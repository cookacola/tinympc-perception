#!/usr/bin/env python3
"""Quantify between-flight appearance and gate-pose shift in real labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gap8_perception.audit_real_flights import canonical_image_order


APPEARANCE_NAMES = (
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_median",
    "intensity_p90",
    "gradient_mean",
)
POSE_NAMES = (
    "gate_center_x",
    "gate_center_y",
    "gate_width",
    "gate_height",
    "gate_area",
    "gate_aspect",
    "top_bottom_width_ratio",
    "left_right_height_ratio",
)
FEATURE_NAMES = APPEARANCE_NAMES + POSE_NAMES


def polygon_area(points: np.ndarray) -> float:
    return float(
        0.5
        * abs(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )
    )


def frame_features(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    crop = image[20:140].astype(np.float32) / 255.0
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    tl, tr, br, bl = canonical_image_order(corners)[0]
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return np.asarray(
        [
            crop.mean(),
            crop.std(),
            *np.percentile(crop, (10, 50, 90)),
            np.hypot(gx, gy).mean(),
            corners[:, 0].mean() / 160.0,
            (corners[:, 1].mean() - 20.0) / 120.0,
            width / 160.0,
            height / 120.0,
            polygon_area(np.asarray((tl, tr, br, bl))) / (160.0 * 120.0),
            width / max(height, 1e-6),
            np.linalg.norm(tr - tl) / max(np.linalg.norm(br - bl), 1e-6),
            np.linalg.norm(bl - tl) / max(np.linalg.norm(br - tr), 1e-6),
        ],
        np.float64,
    )


def temporal_centroid_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_indices: np.ndarray,
    columns: slice,
    folds: int = 5,
) -> float:
    values = features[:, columns]
    predictions, truths = [], []
    for fold in range(folds):
        test = (sequence_indices % folds) == fold
        train = ~test
        mean = values[train].mean(0)
        scale = values[train].std(0).clip(1e-6)
        normalized = (values - mean) / scale
        centroids = np.stack(
            [normalized[train & (labels == label)].mean(0) for label in range(3)]
        )
        distance = ((normalized[test, None] - centroids[None]) ** 2).sum(2)
        predictions.extend(distance.argmin(1).tolist())
        truths.extend(labels[test].tolist())
    return float(np.mean(np.asarray(predictions) == np.asarray(truths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/cchen/real_flight_data")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flights = ("flight_06", "flight_07", "flight_08")
    feature_rows, label_rows, sequence_rows = [], [], []
    per_flight = {}
    for label, flight in enumerate(flights):
        folder = args.root / flight
        rows = [
            json.loads(line)
            for line in (folder / "labels.jsonl").read_text().splitlines()
            if line
        ]
        flight_features = []
        kept_index = 0
        for row in rows:
            corners = canonical_image_order(
                np.asarray(row["corners"], np.float32).reshape(4, 2)
            )[0]
            if (corners[:, 1] < 20).any() or (corners[:, 1] >= 140).any():
                continue
            image = cv2.imread(
                str(folder / "stream_out" / row["image"]),
                cv2.IMREAD_GRAYSCALE,
            )
            values = frame_features(image, corners)
            flight_features.append(values)
            feature_rows.append(values)
            label_rows.append(label)
            # Five interleaved temporal blocks avoid testing an immediate
            # neighbor of every training frame.
            sequence_rows.append((kept_index * 5) // max(len(rows), 1))
            kept_index += 1
        values = np.asarray(flight_features)
        per_flight[flight] = {
            "frames": len(values),
            "feature_quantiles": {
                name: {
                    "p10": float(np.percentile(values[:, index], 10)),
                    "median": float(np.median(values[:, index])),
                    "p90": float(np.percentile(values[:, index], 90)),
                }
                for index, name in enumerate(FEATURE_NAMES)
            },
        }
    features = np.asarray(feature_rows)
    labels = np.asarray(label_rows)
    sequence_indices = np.asarray(sequence_rows)
    pooled_scale = features.std(0).clip(1e-6)
    flight_means = np.stack(
        [features[labels == label].mean(0) for label in range(3)]
    )
    standardized_differences = np.max(
        np.abs(flight_means[:, None] - flight_means[None, :])
        / pooled_scale[None, None],
        axis=(0, 1),
    )
    report = {
        "root": str(args.root),
        "scope": "labeled, crop-valid, gate-positive frames only",
        "flights": per_flight,
        "between_flight_shift": {
            "chance_flight_identification_accuracy": 1.0 / 3.0,
            "appearance_only_temporal_block_accuracy": temporal_centroid_accuracy(
                features, labels, sequence_indices, slice(0, len(APPEARANCE_NAMES))
            ),
            "pose_only_temporal_block_accuracy": temporal_centroid_accuracy(
                features, labels, sequence_indices, slice(len(APPEARANCE_NAMES), None)
            ),
            "all_features_temporal_block_accuracy": temporal_centroid_accuracy(
                features, labels, sequence_indices, slice(None)
            ),
            "maximum_pairwise_standardized_mean_difference": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, standardized_differences)
            },
        },
        "interpretation": (
            "Accuracy above chance shows that appearance and/or gate-pose "
            "statistics identify the source flight despite temporal blocking. "
            "This is evidence of domain shift, not a causal explanation."
        ),
        "mitigation": {
            "implemented": (
                "whole-flight splits; configurable training-only geometric, "
                "exposure, gamma, noise, blur, and reflection augmentation"
            ),
            "selection_status": (
                "augmentation is disabled by default because strong and mild "
                "candidates improved flight 07 but regressed untouched flight "
                "08; they were audited and rejected"
            ),
            "still_required": (
                "collect independent tracks, lighting conditions, negative "
                "gate frames, and real obstacle/collision labels"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
