#!/usr/bin/env python3
"""Controlled heatmap-versus-coordinate corner-head training ablation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gap8_perception.data import MultiTaskDataset
from gap8_perception.evaluate import local_centroid
from gap8_perception.model import DSConv, Gap8MultiTaskNet
from gap8_perception.losses import weighted_corner_mse


class CornerAblationNet(nn.Module):
    """Identical encoder with either four heatmaps or eight coordinates."""

    def __init__(self, head: str):
        super().__init__()
        if head not in {"heatmap", "direct"}:
            raise ValueError(head)
        base = Gap8MultiTaskNet()
        self.stem = base.stem
        self.e1_down = base.e1_down
        self.e1_refine = base.e1_refine
        self.geometry40 = base.geometry40
        self.head_kind = head
        channels = 4 if head == "heatmap" else 8
        self.head = nn.Sequential(DSConv(16, 12), nn.Conv2d(12, channels, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.stem(image)
        feature = self.e1_refine(self.e1_down(feature))
        output = self.head(self.geometry40(feature))
        if self.head_kind == "direct":
            output = F.adaptive_avg_pool2d(output, 1).flatten(1).reshape(-1, 4, 2)
        return output


def prediction_and_loss(model, batch):
    output = model(batch["image"])
    valid = batch["corner_valid"]
    if model.head_kind == "heatmap":
        loss = weighted_corner_mse(output, batch["corners"], valid)
        prediction = local_centroid(output.sigmoid()) * 4.0
    else:
        normalized = output.sigmoid()
        loss = output.sum() * 0.0
        if valid.any():
            loss = F.smooth_l1_loss(
                normalized[valid], batch["corner_xy"][valid] / 159.0
            )
        prediction = normalized * 159.0
    return loss, prediction


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    errors = []
    all4_at5 = []
    all4_at10 = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, prediction = prediction_and_loss(model, batch)
            if training:
                loss.backward()
                optimizer.step()
            size = batch["image"].shape[0]
            loss_sum += float(loss.detach()) * size
            sample_count += size
            valid = batch["corner_valid"]
            if valid.any():
                point_error = torch.linalg.vector_norm(
                    prediction[valid] - batch["corner_xy"][valid], dim=2
                )
                errors.append(point_error.detach().cpu())
                all4_at5.append((point_error <= 5.0).all(1).detach().cpu())
                all4_at10.append((point_error <= 10.0).all(1).detach().cpu())
    point_errors = torch.cat(errors).numpy() if errors else np.asarray([], np.float32)
    return {
        "loss": loss_sum / max(1, sample_count),
        "valid_images": int(sum(len(values) for values in all4_at5)),
        "mean_corner_error_px": float(point_errors.mean()) if point_errors.size else float("nan"),
        "median_corner_error_px": float(np.median(point_errors)) if point_errors.size else float("nan"),
        "p95_corner_error_px": float(np.percentile(point_errors, 95)) if point_errors.size else float("nan"),
        "all4_within_5px": float(torch.cat(all4_at5).float().mean()) if all4_at5 else float("nan"),
        "all4_within_10px": float(torch.cat(all4_at10).float().mean()) if all4_at10 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=("heatmap", "direct"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = MultiTaskDataset(args.dataset, args.targets, args.split_file, "train")
    validation = MultiTaskDataset(args.dataset, args.targets, args.split_file, "validation")
    loaders = {
        "train": DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0),
        "validation": DataLoader(validation, args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=device.type == "cuda", persistent_workers=args.workers > 0),
    }
    model = CornerAblationNet(args.head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    log_path = args.output / "log.csv"
    fields = ["epoch", "seconds", "train_loss", "val_loss", "mean_corner_error_px",
              "median_corner_error_px", "p95_corner_error_px", "all4_within_5px",
              "all4_within_10px", "valid_images"]
    best = float("inf")
    with log_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            train_metrics = run_epoch(model, loaders["train"], device, optimizer)
            val_metrics = run_epoch(model, loaders["validation"], device)
            row = {"epoch": epoch, "seconds": time.time() - started,
                   "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]}
            row.update({key: val_metrics[key] for key in fields[4:]})
            writer.writerow(row)
            stream.flush()
            print(json.dumps({"head": args.head, "epoch": epoch,
                              "train": train_metrics, "validation": val_metrics}), flush=True)
            state = {"epoch": epoch, "head": args.head, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "validation": val_metrics,
                     "input": [1, 160, 160], "corner_order": ["tl", "tr", "br", "bl"]}
            torch.save(state, args.output / "last.pt")
            if val_metrics["mean_corner_error_px"] < best:
                best = val_metrics["mean_corner_error_px"]
                torch.save(state, args.output / "best.pt")
    (args.output / "summary.json").write_text(json.dumps({"head": args.head,
        "best_mean_corner_error_px": best}, indent=2) + "\n")


if __name__ == "__main__":
    main()
