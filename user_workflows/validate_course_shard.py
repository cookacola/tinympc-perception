#!/usr/bin/env python3
"""Validate count, shape, alignment, metric depth, semantics, and pose diversity."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=Path, required=True)
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--allow-no-gates", action="store_true")
args = parser.parse_args()
MINIMUM_USEFUL_MEAN = 24.0


def numbered(pattern):
    return sorted(args.shard.glob(pattern))


rgb_files = numbered("rgb_*.png")
depth_files = numbered("distance_to_image_plane_*.npy")
semantic_files = numbered("semantic_segmentation_[0-9][0-9][0-9][0-9].png")
poses = [json.loads(line) for line in (args.shard / "poses.jsonl").read_text().splitlines()]
errors = []
for name, files in (
    ("rgb", rgb_files),
    ("depth", depth_files),
    ("semantic", semantic_files),
):
    if len(files) != args.expected:
        errors.append(f"{name} count={len(files)} expected={args.expected}")
if len(poses) != args.expected:
    errors.append(f"pose count={len(poses)} expected={args.expected}")

stats = []
observed_classes = set()
for index, (rgb_path, depth_path, semantic_path) in enumerate(
    zip(rgb_files, depth_files, semantic_files)
):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = np.load(depth_path, allow_pickle=False)
    semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        errors.append(f"{index}: unreadable RGB")
        continue
    rgb_mean = float(rgb.mean())
    rgb_variance = float(rgb.var())
    if rgb_mean <= MINIMUM_USEFUL_MEAN or rgb_variance <= 15:
        errors.append(
            f"{index}: unusable RGB mean={rgb_mean:.2f} variance={rgb_variance:.2f}"
        )
    height, width = rgb.shape[:2]
    if depth.shape[:2] != (height, width):
        errors.append(f"{index}: depth shape {depth.shape} != {(height, width)}")
    if semantic.shape[:2] != (height, width):
        errors.append(f"{index}: semantic shape {semantic.shape} != {(height, width)}")
    valid_depth = depth[np.isfinite(depth) & (depth > 0) & (depth <= 65.535)]
    if valid_depth.size == 0:
        errors.append(f"{index}: no representable positive metric depth")
    ids = np.unique(semantic)
    labels_path = semantic_path.with_name(
        semantic_path.name.replace("semantic_segmentation_", "semantic_segmentation_labels_")
    ).with_suffix(".json")
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        observed_classes.update(
            value.get("class", "").lower() for value in labels.values() if value.get("class")
        )
    stats.append(
        {
            "frame": index,
            "rgb_mean": rgb_mean,
            "rgb_variance": rgb_variance,
            "depth_min_m": float(valid_depth.min()) if valid_depth.size else None,
            "depth_max_m": float(valid_depth.max()) if valid_depth.size else None,
            "semantic_ids": [int(v) for v in ids],
        }
    )

eyes = np.asarray([pose["eye_m"] for pose in poses], dtype=np.float64)
if len(eyes) > 1 and np.min(np.ptp(eyes, axis=0)) < 0.1:
    errors.append("camera poses lack diversity")
required_classes = {"course", "boundary", "obstacle"}
if not args.allow_no_gates:
    required_classes.add("gate")
missing_classes = sorted(required_classes - observed_classes)
if missing_classes:
    errors.append(f"semantic classes absent from shard: {missing_classes}")

report = {
    "passed": not errors,
    "expected": args.expected,
    "validated": len(stats),
    "errors": errors,
    "camera_eye_range_m": np.ptp(eyes, axis=0).tolist() if len(eyes) else [],
    "observed_classes": sorted(observed_classes),
    "frames": stats,
}
(args.shard / "validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("passed", "validated", "errors", "camera_eye_range_m")}, indent=2))
raise SystemExit(0 if not errors else 1)
