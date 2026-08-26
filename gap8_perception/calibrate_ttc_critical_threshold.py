#!/usr/bin/env python3
"""Calibrate critical-risk thresholds on validation and freeze them for test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_dory_model import load_dory_checkpoint


def collect(model, loader, device):
    probabilities, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["onboard_state"].to(device, non_blocking=True)
            output = model(images, state)
            probability = torch.softmax(output["risk_logits"], 1)[:, 2]
            valid = batch["ttc_valid"].bool().squeeze(1)
            critical = (
                batch["ttc_approaching"].bool().squeeze(1)
                & (batch["inverse_ttc"].squeeze(1) >= 2.0)
            )
            probabilities.append(probability.cpu()[valid].numpy())
            labels.append(critical[valid].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def metrics(probability, truth, threshold):
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def calibrate(probability, truth, precision_targets):
    order = np.argsort(-probability, kind="stable")
    sorted_truth = truth[order].astype(np.int64)
    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(1 - sorted_truth)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.count_nonzero(truth)), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    result = {
        "fixed_0_552": metrics(probability, truth, 0.552),
        "maximum_f1": metrics(probability, truth, probability[order[int(np.argmax(f1))]]),
        "precision_constrained": {},
    }
    for target in precision_targets:
        eligible = np.flatnonzero(precision >= target)
        if eligible.size == 0:
            result["precision_constrained"][str(target)] = None
            continue
        index = int(eligible[np.argmax(recall[eligible])])
        result["precision_constrained"][str(target)] = metrics(
            probability, truth, probability[order[index]]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision-targets", type=float, nargs="+", default=(0.60, 0.65, 0.70))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this allocation")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    model, initialization = load_dory_checkpoint(args.checkpoint, device)
    options = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    validation_probability, validation_truth = collect(
        model, DataLoader(TTCGateDataset(args.dataset, "validation"), **options), device
    )
    test_probability, test_truth = collect(
        model, DataLoader(TTCGateDataset(args.dataset, "test"), **options), device
    )
    validation = calibrate(validation_probability, validation_truth, args.precision_targets)
    frozen_test = {
        "fixed_0_552": metrics(test_probability, test_truth, 0.552),
        "maximum_f1_validation_threshold": metrics(
            test_probability, test_truth, validation["maximum_f1"]["threshold"]
        ),
        "validation_precision_constrained_thresholds": {},
    }
    for target, operating_point in validation["precision_constrained"].items():
        frozen_test["validation_precision_constrained_thresholds"][target] = (
            None if operating_point is None else metrics(
                test_probability, test_truth, operating_point["threshold"]
            )
        )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization": initialization,
        "critical_definition": "valid and approaching and inverse_ttc>=2.0_s^-1 (TTC<=0.5s)",
        "calibration_split": "validation",
        "test_thresholds_are_frozen_from_validation": True,
        "validation_pixels": int(validation_truth.size),
        "validation_critical_pixels": int(np.count_nonzero(validation_truth)),
        "test_pixels": int(test_truth.size),
        "test_critical_pixels": int(np.count_nonzero(test_truth)),
        "validation": validation,
        "test": frozen_test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
