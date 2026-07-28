#!/usr/bin/env python3
"""Float training with checkpoint resume and task-specific best models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap8_perception.data import MultiTaskDataset
from gap8_perception.losses import multitask_loss
from gap8_perception.model import Gap8MultiTaskNet


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    task_keys = ("total", "corner", "danger", "urgency", "uncertainty", "gate")
    totals = {key: 0.0 for key in task_keys}
    count = 0
    gradient_norms = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = multitask_loss(
                model.forward_image(batch["image"]), batch
            )
            if training:
                if not gradient_norms:
                    shared = model.stem[0][0].weight
                    for task in ("corner", "danger", "urgency", "uncertainty", "gate"):
                        gradient = torch.autograd.grad(
                            losses[task],
                            shared,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        gradient_norms[f"grad_{task}"] = (
                            float(gradient.norm().detach()) if gradient is not None else 0.0
                        )
                losses["total"].backward()
                optimizer.step()
            size = batch["image"].shape[0]
            for key in totals:
                totals[key] += float(losses[key].detach()) * size
            count += size
    return {key: value / count for key, value in totals.items()} | gradient_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--gate-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--high-speed-stride",
        type=int,
        default=1,
        help="Keep the highest-speed state variant for one in every N training images.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = MultiTaskDataset(
        args.dataset,
        args.targets,
        args.split_file,
        "train",
        args.train_limit,
        augment=args.augment,
        high_speed_stride=args.high_speed_stride,
    )
    validation = MultiTaskDataset(
        args.dataset, args.targets, args.split_file, "validation", args.val_limit
    )
    train_loader = DataLoader(
        train,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        validation,
        args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = Gap8MultiTaskNet(args.gate_head).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    start_epoch = 1
    best = {"total": float("inf"), "corner": float("inf"), "danger": float("inf")}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best.update(state["best"])

    log_path = args.output / "log.csv"
    new_log = not log_path.exists() or start_epoch == 1
    stream = log_path.open("w" if new_log else "a", newline="")
    writer = csv.writer(stream)
    if new_log:
        writer.writerow(
            ["epoch", "seconds"]
            + [f"train_{k}" for k in ("total", "corner", "danger", "urgency", "uncertainty", "gate")]
            + [f"train_grad_{k}" for k in ("corner", "danger", "urgency", "uncertainty", "gate")]
            + [f"val_{k}" for k in ("total", "corner", "danger", "urgency", "uncertainty", "gate")]
        )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        seconds = time.time() - started
        row = (
            [epoch, seconds]
            + [tr[k] for k in ("total", "corner", "danger", "urgency", "uncertainty", "gate")]
            + [tr.get(f"grad_{k}", 0.0) for k in ("corner", "danger", "urgency", "uncertainty", "gate")]
            + [va[k] for k in ("total", "corner", "danger", "urgency", "uncertainty", "gate")]
        )
        writer.writerow(row)
        stream.flush()
        print(json.dumps({"epoch": epoch, "seconds": seconds, "train": tr, "validation": va}), flush=True)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
            "architecture": "Gap8MultiTaskNet",
            "gate_head": args.gate_head,
            "state_dim": model.state_dim,
            "motion_conditioning": "controller_side_range_postprocessor",
            "network_inputs": ["hm01b0"],
            "input": [1, 160, 160],
        }
        torch.save(state, args.output / "last.pt")
        for metric in best:
            if va[metric] < best[metric]:
                best[metric] = va[metric]
                state["best"] = best.copy()
                torch.save(state, args.output / f"best_{metric}.pt")
    stream.close()


if __name__ == "__main__":
    main()
