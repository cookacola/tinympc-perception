#!/usr/bin/env python3
"""Mixed synthetic/full-task and real/corner-only domain adaptation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.data_stdc_real import RealCornerDataset
from gap8_perception.losses_stdc import design_multitask_loss, focal_heatmap_mse
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet, ProposedSTDCFPNNet
from gap8_perception.train_stdc import move, run_epoch


def run_mixed(model, synthetic_loader, real_loader, device, optimizer, real_weight):
    model.train()
    real_iterator = itertools.cycle(real_loader)
    totals = {"total": 0.0, "synthetic": 0.0, "real_corner": 0.0}
    count = 0
    for synthetic in synthetic_loader:
        synthetic = move(synthetic, device)
        real = move(next(real_iterator), device)
        optimizer.zero_grad(set_to_none=True)
        synthetic_loss = design_multitask_loss(
            model(synthetic["image"]), synthetic
        )["total"]
        real_outputs = model(real["image"])
        real_corner = focal_heatmap_mse(
            real_outputs["corners"], real["corners"], real["corner_valid"]
        )
        total = synthetic_loss + real_weight * real_corner
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        size = synthetic["image"].shape[0]
        totals["total"] += float(total.detach()) * size
        totals["synthetic"] += float(synthetic_loss.detach()) * size
        totals["real_corner"] += float(real_corner.detach()) * size
        count += size
    return {key: value / count for key, value in totals.items()}


def run_real_corner(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            outputs = model.predict(batch["image"])
            loss = focal_heatmap_mse(
                torch.logit(outputs["corners"].clamp(1e-6, 1.0 - 1e-6)),
                batch["corners"],
                batch["corner_valid"],
            )
            probability = outputs["corners"]
            batch_size, channels, height, width = probability.shape
            flat = probability.flatten(2).argmax(2)
            pred_y = torch.div(flat, width, rounding_mode="floor").float()
            pred_x = (flat % width).float()
            prediction = torch.stack(
                (pred_x * 4.0 + 2.0, pred_y * 4.0 + 22.0), dim=2
            )
            error = torch.linalg.vector_norm(
                prediction - batch["corner_xy"], dim=2
            ).mean(1)
            errors.extend(error.cpu().tolist())
            total += float(loss) * batch_size
            count += batch_size
    return {
        "loss": total / max(count, 1),
        "mean_error_px": sum(errors) / max(len(errors), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--real-weight", type=float, default=2.0)
    parser.add_argument("--real-augmentation-strength", type=float, default=0.35)
    parser.add_argument(
        "--real-augmentation-probability", type=float, default=0.0
    )
    parser.add_argument("--real-train-flights", default="flight_06")
    parser.add_argument("--real-validation-flights", default="flight_07")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = (
        ProposedSTDCFPNNet()
        if state.get("architecture") == "ProposedSTDCFPNNet"
        else Gap8STDCMultiHeadNet()
    )
    model.load_state_dict(state["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    synthetic = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "train", augment=True
    )
    validation = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "validation"
    )
    real_train_flights = tuple(
        item for item in args.real_train_flights.split(",") if item
    )
    real_validation_flights = tuple(
        item for item in args.real_validation_flights.split(",") if item
    )
    real = RealCornerDataset(
        args.real_root,
        real_train_flights,
        augment=args.real_augmentation_probability > 0.0,
        augmentation_strength=args.real_augmentation_strength,
        augmentation_probability=args.real_augmentation_probability,
    )
    real_validation = RealCornerDataset(args.real_root, real_validation_flights)
    loaders = [
        DataLoader(
            dataset,
            args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for dataset, shuffle in (
            (synthetic, True),
            (real, True),
            (validation, False),
            (real_validation, False),
        )
    ]
    best = float("inf")
    log = (args.output / "log.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(
        [
            "epoch", "train_total", "train_synthetic", "train_real_corner",
            "synthetic_val_total", "synthetic_val_danger",
            "real_val_corner_loss", "real_val_mean_error_px", "selection_score",
        ]
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_mixed(
            model, loaders[0], loaders[1], device, optimizer, args.real_weight
        )
        val_metrics = run_epoch(model, loaders[2], device)
        real_val_metrics = run_real_corner(model, loaders[3], device)
        # Real localization drives adaptation selection while synthetic danger
        # remains a safety regularizer. Flight 08 is never consulted here.
        selection_score = (
            real_val_metrics["loss"] + 0.10 * val_metrics["danger"]
        )
        writer.writerow(
            [
                epoch,
                train_metrics["total"],
                train_metrics["synthetic"],
                train_metrics["real_corner"],
                val_metrics["total"],
                val_metrics["danger"],
                real_val_metrics["loss"],
                real_val_metrics["mean_error_px"],
                selection_score,
            ]
        )
        log.flush()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": type(model).__name__,
            "adaptation": {
                "real_train_flights": list(real_train_flights),
                "real_validation_flights": list(real_validation_flights),
                "held_out_real_flight": "flight_08",
                "real_task": "corners_only",
                "real_weight": args.real_weight,
                "real_augmentation_strength": args.real_augmentation_strength,
                "real_augmentation_probability": (
                    args.real_augmentation_probability
                ),
            },
            "selection_score": selection_score,
            "best_selection_score": min(best, selection_score),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if selection_score < best:
            best = selection_score
            checkpoint["best_selection_score"] = best
            torch.save(checkpoint, args.output / "best_total.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "real_validation": real_val_metrics,
                    "selection_score": selection_score,
                }
            ),
            flush=True,
        )
    log.close()


if __name__ == "__main__":
    main()
