#!/usr/bin/env python3
"""PyTorch fake-quant fine-tuning for the sequential student."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import disable_observer, enable_observer
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .data_sequential import RealSequentialCornerDataset, SequentialTargetDataset
from .losses_sequential import decode_offsets, sequential_loss
from .model_sequential import SequentialSTDCNet
from .quantization import prepare_int8_qat


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "heatmap", "offset", "confidence", "field")}
    count, overestimates = 0, []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"])
            losses = sequential_loss(output, batch)
            loss = losses["total"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch["image"].shape[0]
            with torch.no_grad():
                predicted = decode_offsets(output[:, 4:8].mean(dim=(-2, -1)))
                valid = batch["offset_valid"] > 0.5
                overestimates.extend(
                    (predicted - batch["offset_m"])[valid].detach().cpu().tolist()
                )
            count += batch["image"].shape[0]
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
    parser.add_argument("--float-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
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
    state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=False)
    base = SequentialSTDCNet()
    base.load_state_dict(state["model"])
    model = prepare_int8_qat(base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train = SequentialTargetDataset(
        args.dataset, args.targets, args.split_file, "train",
        observation_dropout_probability=args.observation_dropout,
        sensor_augmentation_probability=args.sensor_augmentation,
    )
    validation = SequentialTargetDataset(args.dataset, args.targets, args.split_file, "validation")
    train_dataset, sampler, real_count = train, None, 0
    if args.real_root and args.real_weight > 0:
        real = RealSequentialCornerDataset(
            args.real_root,
            tuple(item for item in args.real_train_flights.split(",") if item),
            sensor_augmentation_probability=args.sensor_augmentation,
        )
        real_count = len(real)
        train_dataset = ConcatDataset((train, real))
        weights = torch.cat((
            torch.full((len(train),), 1.0 / len(train)),
            torch.full((len(real),), args.real_weight / len(real)),
        ))
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    loaders = [
        DataLoader(train_dataset, args.batch_size, shuffle=sampler is None, sampler=sampler,
                   num_workers=args.workers, pin_memory=device.type == "cuda",
                   persistent_workers=args.workers > 0),
        DataLoader(validation, args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=device.type == "cuda", persistent_workers=args.workers > 0),
    ]
    best = (float("inf"), float("inf"), float("inf"))
    for epoch in range(1, args.epochs + 1):
        model.apply(enable_observer if epoch <= max(1, args.epochs - 3) else disable_observer)
        train_metrics = run_epoch(model, loaders[0], device, optimizer)
        model.apply(disable_observer)
        validation_metrics = run_epoch(model, loaders[1], device)
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "architecture": "SequentialSTDCNet",
            "quantization_aware": True, "backend": "qnnpack", "input": [1, 120, 160],
            "output": [12, 15, 20], "float_checkpoint": str(args.float_checkpoint),
            "train_metrics": train_metrics, "validation_metrics": validation_metrics,
            "label_contract": "sequential_fixed_normal_v1",
            "seed": args.seed, "dataset": str(args.dataset),
            "targets": str(args.targets), "split_file": str(args.split_file),
            "real_root": str(args.real_root) if args.real_root else None,
            "real_train_flights": args.real_train_flights,
            "sensor_augmentation": args.sensor_augmentation,
            "real_training_records": real_count,
            "real_weight": args.real_weight if real_count else 0.0,
        }
        torch.save(checkpoint, args.output / "last_qat.pt")
        selection = (
            validation_metrics["offset_overestimation_p95_m"],
            validation_metrics["offset_overestimation_mean_m"],
            validation_metrics["total"],
        )
        if selection < best:
            best = selection
            torch.save(checkpoint, args.output / "best_qat.pt")
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": validation_metrics}), flush=True)


if __name__ == "__main__":
    main()
