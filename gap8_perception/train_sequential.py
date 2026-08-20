"""Slurm-friendly trainer for the geometry-labeled single-output student."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .data_sequential import RealSequentialCornerDataset, SequentialTargetDataset
from .losses_sequential import sequential_loss
from .losses_sequential import decode_offsets
from .model_sequential import SequentialSTDCNet


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "heatmap", "offset", "confidence", "field")}
    count, overestimates = 0, []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device)
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(image)
            losses = sequential_loss(output, batch)
            loss = losses["total"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * image.shape[0]
            with torch.no_grad():
                predicted = decode_offsets(output[:, 4:8].mean(dim=(-2, -1)))
                valid = batch["offset_valid"] > 0.5
                overestimates.extend(
                    (predicted - batch["offset_m"])[valid].detach().cpu().tolist()
                )
            count += image.shape[0]
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    positive = np.maximum(np.asarray(overestimates, np.float64), 0.0)
    metrics.update({
        "offset_overestimation_mean_m": float(positive.mean()) if len(positive) else 0.0,
        "offset_overestimation_p95_m": float(np.quantile(positive, 0.95)) if len(positive) else 0.0,
        "false_safe_fraction": float((np.asarray(overestimates) > 0).mean()) if overestimates else 0.0,
    })
    return metrics


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
    parser.add_argument("--observation-dropout", type=float, default=0.10)
    parser.add_argument("--sensor-augmentation", type=float, default=1.0)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--real-train-flights", default="flight_06,flight_07")
    parser.add_argument("--real-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = SequentialTargetDataset(
        args.dataset, args.targets, args.split_file, "train", limit=args.train_limit,
        observation_dropout_probability=args.observation_dropout,
        sensor_augmentation_probability=args.sensor_augmentation,
    )
    valid = SequentialTargetDataset(args.dataset, args.targets, args.split_file, "validation", limit=args.val_limit)
    train_dataset = train
    train_sampler = None
    real_count = 0
    if args.real_root and args.real_weight > 0:
        flights = tuple(item for item in args.real_train_flights.split(",") if item)
        real = RealSequentialCornerDataset(
            args.real_root, flights,
            sensor_augmentation_probability=args.sensor_augmentation,
        )
        real_count = len(real)
        train_dataset = ConcatDataset((train, real))
        weights = torch.cat((
            torch.full((len(train),), 1.0 / len(train)),
            torch.full((len(real),), args.real_weight / len(real)),
        ))
        train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    loaders = [
        DataLoader(train_dataset, args.batch_size, shuffle=train_sampler is None,
                   sampler=train_sampler, num_workers=args.workers,
                   pin_memory=device.type == "cuda", persistent_workers=args.workers > 0),
        DataLoader(valid, args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=device.type == "cuda", persistent_workers=args.workers > 0),
    ]
    model = SequentialSTDCNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = (float("inf"), float("inf"), float("inf"))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders[0], device, optimizer)
        valid_metrics = run_epoch(model, loaders[1], device)
        state = {"epoch": epoch, "model": model.state_dict(), "architecture": type(model).__name__,
                 "input": [1, 120, 160], "output": [12, 15, 20],
                 "train_metrics": train_metrics, "validation_metrics": valid_metrics,
                 "label_contract": "sequential_fixed_normal_v1",
                 "observation_dropout": args.observation_dropout,
                 "sensor_augmentation": args.sensor_augmentation,
                 "seed": args.seed,
                 "dataset": str(args.dataset), "targets": str(args.targets),
                 "split_file": str(args.split_file),
                 "real_root": str(args.real_root) if args.real_root else None,
                 "real_train_flights": args.real_train_flights,
                 "real_training_records": real_count,
                 "real_weight": args.real_weight if real_count else 0.0}
        torch.save(state, args.output / "last.pt")
        selection = (
            valid_metrics["offset_overestimation_p95_m"],
            valid_metrics["offset_overestimation_mean_m"],
            valid_metrics["total"],
        )
        if selection < best:
            best = selection
            torch.save(state, args.output / "best.pt")
        metrics = {key: value for key, value in state.items() if key != "model"}
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": valid_metrics}), flush=True)


if __name__ == "__main__":
    main()
