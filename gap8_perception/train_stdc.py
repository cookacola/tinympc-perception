#!/usr/bin/env python3
"""Train the design-document 160x120 STDC multi-head network."""

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

from gap8_perception.data_stdc import STDCMultiTaskDataset, STDCPrivilegedDataset
from gap8_perception.losses_stdc import design_multitask_loss
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet, Gap8STDCPrivilegedTeacher


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, optimizer=None, image_key="image"):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "corner", "danger")}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = design_multitask_loss(model(batch[image_key]), batch)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            size = batch["image"].shape[0]
            for key in totals:
                totals[key] += float(losses[key].detach()) * size
            count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--privileged-teacher",
        action="store_true",
        help="Train a two-channel monochrome+inverse-depth teacher.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_class = STDCPrivilegedDataset if args.privileged_teacher else STDCMultiTaskDataset
    train = dataset_class(
        args.dataset,
        args.targets,
        args.split_file,
        "train",
        args.train_limit,
        augment=args.augment,
    )
    validation = dataset_class(
        args.dataset, args.targets, args.split_file, "validation", args.val_limit
    )
    loaders = [
        DataLoader(
            dataset,
            args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for dataset, shuffle in ((train, True), (validation, False))
    ]
    model = (
        Gap8STDCPrivilegedTeacher()
        if args.privileged_teacher
        else Gap8STDCMultiHeadNet()
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-5
    )
    start_epoch = 1
    best = {"total": float("inf"), "corner": float("inf"), "danger": float("inf")}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best.update(state["best"])

    log_path = args.output / "log.csv"
    stream = log_path.open("a" if start_epoch > 1 else "w", newline="")
    writer = csv.writer(stream)
    if start_epoch == 1:
        writer.writerow(
            ["epoch", "seconds", "learning_rate"]
            + [f"train_{key}" for key in best]
            + [f"val_{key}" for key in best]
        )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        image_key = "privileged_image" if args.privileged_teacher else "image"
        train_metrics = run_epoch(
            model, loaders[0], device, optimizer, image_key=image_key
        )
        val_metrics = run_epoch(model, loaders[1], device, image_key=image_key)
        elapsed = time.time() - started
        writer.writerow(
            [epoch, elapsed, optimizer.param_groups[0]["lr"]]
            + [train_metrics[key] for key in best]
            + [val_metrics[key] for key in best]
        )
        stream.flush()
        scheduler.step()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best.copy(),
            "architecture": (
                "Gap8STDCPrivilegedTeacher"
                if args.privileged_teacher
                else "Gap8STDCMultiHeadNet"
            ),
            "input": [2 if args.privileged_teacher else 1, 120, 160],
            "corner_output": [4, 30, 40],
            "danger_output": [1, 15, 20],
            "crop": {"top": 20, "bottom": 140},
        }
        torch.save(state, args.output / "last.pt")
        for metric in best:
            if val_metrics[metric] < best[metric]:
                best[metric] = val_metrics[metric]
                state["best"] = best.copy()
                torch.save(state, args.output / f"best_{metric}.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seconds": elapsed,
                    "train": train_metrics,
                    "validation": val_metrics,
                }
            ),
            flush=True,
        )
    stream.close()


if __name__ == "__main__":
    main()
