#!/usr/bin/env python3
"""INT8 fake-quantization fine-tuning for Gap8STDCMultiHeadNet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.ao.quantization import disable_observer, enable_observer
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet
from gap8_perception.quantization import prepare_int8_qat
from gap8_perception.train_stdc import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--float-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    float_state = torch.load(
        args.float_checkpoint, map_location="cpu", weights_only=False
    )
    base = Gap8STDCMultiHeadNet()
    base.load_state_dict(float_state["model"])
    model = prepare_int8_qat(base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start, best = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = int(state["epoch"]) + 1, float(state["best_total"])
    train = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "train", augment=True
    )
    validation = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "validation"
    )
    train_loader = DataLoader(
        train, args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        validation, args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    log = (args.output / "log.csv").open("a" if args.resume else "w", newline="")
    writer = csv.writer(log)
    if not args.resume:
        writer.writerow(
            ["epoch", "train_total", "val_total", "val_corner", "val_danger"]
        )
    for epoch in range(start, args.epochs + 1):
        model.apply(enable_observer if epoch <= max(1, args.epochs - 5) else disable_observer)
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        model.apply(disable_observer)
        val_metrics = run_epoch(model, val_loader, device)
        writer.writerow(
            [
                epoch,
                train_metrics["total"],
                val_metrics["total"],
                val_metrics["corner"],
                val_metrics["danger"],
            ]
        )
        log.flush()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_total": min(best, val_metrics["total"]),
            "architecture": "Gap8STDCMultiHeadNet",
            "input": [1, 120, 160],
            "quantization_aware": True,
            "backend": "qnnpack",
        }
        torch.save(state, args.output / "last_qat.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            state["best_total"] = best
            torch.save(state, args.output / "best_qat.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": val_metrics,
                }
            ),
            flush=True,
        )
    log.close()


if __name__ == "__main__":
    main()
