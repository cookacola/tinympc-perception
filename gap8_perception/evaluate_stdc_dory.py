#!/usr/bin/env python3
"""Held-out safety and corner metrics for the resize-free DORY pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.evaluate import binary_counts, local_centroid, safe_div
from gap8_perception.model_stdc_dory import (
    Gap8STDCCornerDoryNet,
    Gap8STDCDangerDoryNet,
)
from gap8_perception.profile_stdc_dory import combined_profile
from gap8_perception.train_stdc_dory_students import conservative_danger_target


def metrics(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return counts | {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "false_negative_rate": safe_div(fn, tp + fn),
        "iou": safe_div(tp, tp + fp + fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    corner, danger = Gap8STDCCornerDoryNet(), Gap8STDCDangerDoryNet()
    corner.load_state_dict(state["corner_model"])
    danger.load_state_dict(state["danger_model"])
    corner, danger = corner.to(device).eval(), danger.to(device).eval()
    dataset = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "test"
    )
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.workers
    )
    errors = []
    gate_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    danger_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            corner_probability = corner(image).sigmoid()
            danger_probability = danger(image).sigmoid().cpu().numpy()
            confidence = corner_probability.flatten(2).amax(2).cpu().numpy()
            valid = batch["corner_valid"].numpy().astype(bool)
            update = binary_counts((confidence >= 0.25).all(axis=1), valid)
            for key in gate_counts:
                gate_counts[key] += update[key]
            prediction = local_centroid(corner_probability).cpu().numpy()
            prediction[..., 0] *= 4.0
            prediction[..., 1] = prediction[..., 1] * 4.0 + 20.0
            if valid.any():
                errors.extend(
                    np.linalg.norm(
                        prediction[valid] - batch["corner_xy"].numpy()[valid],
                        axis=2,
                    ).ravel().tolist()
                )
            truth = (
                conservative_danger_target(batch["danger"]).numpy() >= 0.5
            )
            update = binary_counts(danger_probability >= 0.5, truth)
            for key in danger_counts:
                danger_counts[key] += update[key]
    errors = np.asarray(errors, np.float64)
    report = {
        "checkpoint": str(args.checkpoint),
        "images": len(dataset),
        "resource_profile": combined_profile(),
        "corners": {
            "mean_error_image_px": float(errors.mean()),
            "pck_at_4px": float((errors <= 4.0).mean()),
            "gate_detection_at_0.25": metrics(gate_counts),
        },
        "danger_at_0.5": metrics(danger_counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
