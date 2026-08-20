#!/usr/bin/env python3
"""Held-out metrics and operating-point sweeps for the STDC design model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.evaluate import binary_counts, local_centroid, safe_div
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet, ProposedSTDCFPNNet
from gap8_perception.profile_stdc import profile
from gap8_perception.quantization import prepare_int8_qat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = (
        ProposedSTDCFPNNet()
        if state.get("architecture") == "ProposedSTDCFPNNet"
        else Gap8STDCMultiHeadNet()
    )
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, args.split
    )
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.workers
    )
    confidence_thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    danger_thresholds = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    gate_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in confidence_thresholds
    }
    danger_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in danger_thresholds
    }
    dory_danger_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    corner_errors = []
    danger_mae_sum = 0.0
    danger_items = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model.predict(batch["image"].to(device))
            confidence = outputs["corner_confidence"].cpu().numpy().min(axis=1)
            valid = batch["corner_valid"].numpy().astype(bool)
            for threshold, counts in gate_counts.items():
                update = binary_counts(confidence >= threshold, valid)
                for key in counts:
                    counts[key] += update[key]

            pred_xy = local_centroid(outputs["corners"]).cpu().numpy()
            pred_xy[..., 0] *= 4.0
            pred_xy[..., 1] = pred_xy[..., 1] * 4.0 + 20.0
            if valid.any():
                errors = np.linalg.norm(
                    pred_xy[valid] - batch["corner_xy"].numpy()[valid], axis=2
                )
                corner_errors.extend(errors.ravel().tolist())

            danger_probability = outputs["danger"].cpu().numpy()
            danger_truth_float = batch[
                "danger_dense" if danger_probability.shape[-2:] == (30, 40) else "danger"
            ].numpy()
            danger_truth = danger_truth_float >= 0.5
            danger_mae_sum += float(
                np.abs(danger_probability - danger_truth_float).sum()
            )
            danger_items += danger_truth_float.size
            for threshold, counts in danger_counts.items():
                update = binary_counts(danger_probability >= threshold, danger_truth)
                for key in counts:
                    counts[key] += update[key]
            # The current release's DORY danger graph is 8x10 and uses an
            # adaptive max-pooled safety target.  Report this projection too,
            # so the proposed dense danger head is compared at exactly the
            # same operating resolution and conservative target definition.
            dory_probability = torch.nn.functional.adaptive_max_pool2d(
                torch.from_numpy(danger_probability), (8, 10)
            ).numpy()
            dory_truth = torch.nn.functional.adaptive_max_pool2d(
                batch["danger"], (8, 10)
            ).numpy() >= 0.5
            update = binary_counts(dory_probability >= 0.5, dory_truth)
            for key in dory_danger_counts:
                dory_danger_counts[key] += update[key]

    def classification(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return counts | {
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "false_negative_rate": safe_div(fn, tp + fn),
            "iou": safe_div(tp, tp + fp + fn),
        }

    errors = np.asarray(corner_errors, dtype=np.float64)
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "images": len(dataset),
        "architecture": profile(model),
        "corners": {
            "mean_error_image_px": float(errors.mean()) if len(errors) else None,
            "median_error_image_px": float(np.median(errors)) if len(errors) else None,
            "pck_at_4px": float((errors <= 4.0).mean()) if len(errors) else None,
            "confidence_sweep": {
                str(key): classification(value) for key, value in gate_counts.items()
            },
        },
        "danger": {
            "probability_mae": danger_mae_sum / max(danger_items, 1),
            "threshold_sweep": {
                str(key): classification(value)
                for key, value in danger_counts.items()
            },
        },
        "danger_dory_compatible_at_0.5": classification(dory_danger_counts),
    }
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text())
        baseline_corner = baseline["corners"]
        baseline_danger = baseline.get("danger_at_0.5")
        if baseline_danger is None:
            baseline_danger = baseline["danger"]["threshold_sweep"]["0.5"]
        candidate_gate = report["corners"]["confidence_sweep"]["0.25"]
        candidate_danger = report["danger_dory_compatible_at_0.5"]
        report["comparison"] = {
            "baseline_report": str(args.baseline_report),
            "baseline_checkpoint": baseline.get("checkpoint"),
            "delta_candidate_minus_baseline": {
                "corner_mean_error_image_px": (
                    report["corners"]["mean_error_image_px"]
                    - baseline_corner["mean_error_image_px"]
                ),
                "corner_pck_at_4px": (
                    report["corners"]["pck_at_4px"]
                    - baseline_corner["pck_at_4px"]
                ),
                "gate_recall_at_0.25": (
                    candidate_gate["recall"]
                    - baseline_corner["gate_detection_at_0.25"]["recall"]
                ),
                "danger_recall_at_0.5": (
                    candidate_danger["recall"] - baseline_danger["recall"]
                ),
                "danger_iou_at_0.5": (
                    candidate_danger["iou"] - baseline_danger["iou"]
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
