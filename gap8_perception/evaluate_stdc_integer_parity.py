#!/usr/bin/env python3
"""Label-aware held-out evaluation of decoded NeMO integer outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.evaluate import binary_counts, local_centroid, safe_div
from gap8_perception.train_stdc_dory_students import conservative_danger_target


def classification_metrics(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return counts | {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "false_negative_rate": safe_div(fn, tp + fn),
        "iou": safe_div(tp, tp + fp + fn),
    }


def add_counts(total, prediction, truth):
    update = binary_counts(prediction, truth)
    for key in total:
        total[key] += update[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--integer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "test"
    )
    by_path = {str(record[3]): index for index, record in enumerate(dataset.records)}
    corner_path = args.integer_dir / "corner/corner_parity_predictions.npz"
    danger_path = args.integer_dir / "danger/danger_parity_predictions.npz"
    if not corner_path.is_file():
        corner_path = (
            args.integer_dir
            / "corner_head/corner_head_parity_predictions.npz"
        )
    if not danger_path.is_file():
        danger_path = (
            args.integer_dir
            / "danger_head/danger_head_parity_predictions.npz"
        )
    corner = np.load(corner_path)
    danger = np.load(danger_path)
    if not np.array_equal(corner["paths"], danger["paths"]):
        raise RuntimeError("corner and danger parity path sets differ")

    gate_counts = {
        kind: {key: 0 for key in ("tp", "fp", "fn", "tn")}
        for kind in ("float", "integer")
    }
    danger_counts = {
        kind: {key: 0 for key in ("tp", "fp", "fn", "tn")}
        for kind in ("float", "integer")
    }
    corner_errors = {kind: [] for kind in ("float", "integer")}
    gate_disagreement = 0
    danger_disagreement = 0
    integer_false_safe_vs_float = 0
    integer_danger_probabilities = []
    danger_truths = []

    for row, path in enumerate(corner["paths"]):
        sample = dataset[by_path[str(path)]]
        valid = bool(sample["corner_valid"])
        decisions = {}
        for kind, key in (("float", "float_logits"), ("integer", "integer_logits")):
            probability = torch.from_numpy(corner[key][row]).sigmoid().unsqueeze(0)
            confidence = probability.flatten(2).amax(2).numpy()[0]
            decisions[kind] = bool((confidence >= 0.25).all())
            add_counts(
                gate_counts[kind],
                np.asarray([decisions[kind]]),
                np.asarray([valid]),
            )
            if valid:
                prediction = local_centroid(probability).numpy()[0]
                prediction[:, 0] *= 4.0
                prediction[:, 1] = prediction[:, 1] * 4.0 + 20.0
                corner_errors[kind].extend(
                    np.linalg.norm(
                        prediction - sample["corner_xy"].numpy(), axis=1
                    ).tolist()
                )
        gate_disagreement += decisions["float"] != decisions["integer"]

        truth = (
            conservative_danger_target(sample["danger"].unsqueeze(0)).numpy()
            >= 0.5
        )
        danger_truths.append(truth.ravel())
        danger_decisions = {}
        for kind, key in (("float", "float_logits"), ("integer", "integer_logits")):
            decision = 1.0 / (1.0 + np.exp(-np.clip(danger[key][row], -30, 30))) >= 0.5
            if kind == "integer":
                integer_danger_probabilities.append(
                    1.0
                    / (
                        1.0
                        + np.exp(-np.clip(danger[key][row], -30, 30))
                    ).ravel()
                )
            danger_decisions[kind] = decision
            add_counts(danger_counts[kind], decision[None], truth)
        danger_disagreement += int(
            np.count_nonzero(danger_decisions["float"] != danger_decisions["integer"])
        )
        integer_false_safe_vs_float += int(
            np.count_nonzero(
                danger_decisions["float"] & ~danger_decisions["integer"]
            )
        )

    probabilities = np.concatenate(integer_danger_probabilities)
    truths = np.concatenate(danger_truths)
    threshold_candidates = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], probabilities))
    )
    selected = None
    for threshold in threshold_candidates[::-1]:
        prediction = probabilities >= threshold
        counts = binary_counts(prediction, truths)
        candidate = classification_metrics(counts)
        if candidate["recall"] >= 0.99:
            selected = {"threshold": float(threshold)} | candidate
            break
    if selected is None:
        raise RuntimeError("integer danger output cannot reach 0.99 recall")

    report = {
        "images": len(corner["paths"]),
        "held_out_split": "test",
        "corners": {},
        "danger_at_0.5": {
            kind: classification_metrics(danger_counts[kind])
            for kind in ("float", "integer")
        },
        "recommended_integer_danger_threshold_for_recall_0.99": selected,
        "threshold_recommendation_authoritative": True,
        "gate_decision_disagreements": gate_disagreement,
        "danger_cell_disagreements": danger_disagreement,
        "integer_false_safe_cells_vs_float": integer_false_safe_vs_float,
    }
    for kind in ("float", "integer"):
        errors = np.asarray(corner_errors[kind], np.float64)
        report["corners"][kind] = {
            "mean_error_image_px": float(errors.mean()),
            "pck_at_4px": float((errors <= 4.0).mean()),
            "gate_detection_at_0.25": classification_metrics(gate_counts[kind]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
