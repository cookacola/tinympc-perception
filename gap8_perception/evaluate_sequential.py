#!/usr/bin/env python3
"""Decoded test metrics for the sequential 12-channel student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_sequential import SequentialTargetDataset
from .losses_sequential import decode_offsets
from .model_sequential import SequentialSTDCNet
from .quantization import prepare_int8_qat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SequentialSTDCNet()
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model = model.to(device).eval()
    model.load_state_dict(state["model"])
    dataset = SequentialTargetDataset(args.dataset, args.targets, args.split_file, args.split)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    corner_errors, offset_errors, overestimates, fields = [], [], [], []
    confidence_scores, confidence_targets = [], []
    direction_correct, direction_total = 0, 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            heatmaps = output[:, :4].cpu().numpy()
            truth = batch["corner_xy_crop"].numpy()
            visible_corner = batch["corner_visibility"].numpy().astype(bool)
            for index in range(len(heatmaps)):
                for channel in np.flatnonzero(visible_corner[index]):
                    y, x = np.unravel_index(np.argmax(heatmaps[index, channel]), (15, 20))
                    predicted = np.asarray((8.0 * (x + 0.5) - 0.5, 8.0 * (y + 0.5) - 0.5))
                    corner_errors.append(float(np.linalg.norm(predicted - truth[index, channel])))
            predicted_offsets = decode_offsets(output[:, 4:8].mean(dim=(-2, -1))).cpu()
            error = (predicted_offsets - batch["offset_m"]).numpy()
            mask = batch["offset_valid"].numpy() > 0.5
            offset_errors.extend(np.abs(error[mask]).tolist())
            overestimates.extend(error[mask].tolist())
            confidence_scores.append(
                output[:, 8:12].mean(dim=(-2, -1)).sigmoid().cpu().numpy()
            )
            confidence_targets.append(batch["offset_valid"].numpy())
            predicted_np = predicted_offsets.numpy()
            truth_np = batch["offset_m"].numpy()
            for index in range(len(predicted_np)):
                valid_directions = np.flatnonzero(mask[index])
                if len(valid_directions):
                    direction_correct += int(
                        valid_directions[np.argmax(predicted_np[index, valid_directions])]
                        == valid_directions[np.argmax(truth_np[index, valid_directions])]
                    )
                    direction_total += 1
            fields.append(output[:, 4:12].var(dim=(-2, -1), unbiased=False).cpu().numpy())
    overestimates = np.asarray(overestimates)
    confidence_scores = np.concatenate(confidence_scores).ravel()
    confidence_targets = np.concatenate(confidence_targets).ravel()
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (confidence_scores >= low) & (confidence_scores < high)
        if selected.any():
            ece += selected.mean() * abs(
                confidence_scores[selected].mean() - confidence_targets[selected].mean()
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "quantization_aware": bool(state.get("quantization_aware")),
        "split": args.split,
        "records": len(dataset),
        "corner_error_px": {
            "mean": float(np.mean(corner_errors)),
            "p95": float(np.quantile(corner_errors, 0.95)),
            "count": len(corner_errors),
        },
        "offset_error_m": {
            "mae": float(np.mean(offset_errors)),
            "overestimation_mean": float(np.maximum(overestimates, 0).mean()),
            "overestimation_p95": float(np.quantile(np.maximum(overestimates, 0), 0.95)),
            "overestimation_p99": float(np.quantile(np.maximum(overestimates, 0), 0.99)),
            "false_safe_fraction": float((overestimates > 0).mean()),
            "count": len(offset_errors),
        },
        "direction_selection_accuracy": float(direction_correct / max(direction_total, 1)),
        "confidence_ece_10bin": float(ece),
        "scalar_field_variance": float(np.concatenate(fields).mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
