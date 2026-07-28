#!/usr/bin/env python3
"""Exhaustive 50k audit for duplicates, depth, semantics, gates, and poses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_hashes, pose_hashes, label_hashes = Counter(), Counter(), Counter()
    class_pixels = Counter()
    frames = depth_pixels = invalid_depth = 0
    all_invalid_depth = gate_visible = gate_partial_or_occluded = gate_opening = 0
    missing = []
    global_indices = []
    depth_min_mm, depth_max_mm = 65535, 0
    target_schema_errors = []
    gate_texture_bright_fractions = []
    gate_texture_contrasts = []
    scene_metadata_errors = []

    for shard in sorted(args.dataset.glob("shard_*")):
        if not (shard / "_SUCCESS").is_file():
            continue
        poses = [
            json.loads(line)
            for line in (shard / "poses.jsonl").read_text().splitlines()
        ]
        scene_metadata = json.loads((shard / "scene_metadata.json").read_text())
        texture_version = scene_metadata.get("gate_texture", {}).get("version")
        if texture_version != "newbeedrone_hm01b0_v2":
            scene_metadata_errors.append(
                f"{shard.name}: gate texture is {texture_version!r}"
            )
        targets = np.load(args.targets / f"{shard.name}.npz")
        required_target_shapes = {
            "obstacle_presence_u8": (len(poses), 20, 20),
            "inverse_range_u8": (len(poses), 20, 20),
            "conservative_range_m_f16": (len(poses), 20, 20),
            "danger_u8": (len(poses), 3, 20, 20),
            "time_to_collision_s_f16": (len(poses), 3, 20, 20),
        }
        for key, expected in required_target_shapes.items():
            if key not in targets or targets[key].shape != expected:
                actual = None if key not in targets else targets[key].shape
                target_schema_errors.append(
                    f"{shard.name}:{key}: expected {expected}, got {actual}"
                )
        if (
            "inverse_range_u8" in targets
            and "conservative_range_m_f16" in targets
        ):
            reconstructed = (
                1.0
                - np.clip(
                    targets["conservative_range_m_f16"].astype(np.float32) / 6.0,
                    0.0,
                    1.0,
                )
            ) * 255.0
            if np.max(np.abs(
                reconstructed - targets["inverse_range_u8"].astype(np.float32)
            )) > 1.5:
                target_schema_errors.append(
                    f"{shard.name}: inverse-range/range inconsistency"
                )
        valid_corner = targets["corner_valid_u8"].astype(bool)
        for local, pose in enumerate(poses):
            suffix = f"{local:04d}"
            paths = {
                "mono": shard / f"hm01b0_mono_{suffix}.png",
                "rgb": shard / f"rgb_{suffix}.png",
                "depth": shard / f"depth_mm_{suffix}.png",
                "semantic": shard / f"semantic_segmentation_{suffix}.png",
                "labels": shard / f"semantic_segmentation_labels_{suffix}.json",
            }
            absent = [str(path) for path in paths.values() if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            mono_bytes = paths["mono"].read_bytes()
            image_hashes[hashlib.sha256(mono_bytes).hexdigest()] += 1
            pose_key = json.dumps(
                [pose["eye_m"], pose["target_m"]], separators=(",", ":")
            )
            pose_hashes[hashlib.sha256(pose_key.encode()).hexdigest()] += 1
            labels_text = paths["labels"].read_text()
            label_hashes[hashlib.sha256(labels_text.encode()).hexdigest()] += 1
            labels = json.loads(labels_text)
            semantic = cv2.imread(str(paths["semantic"]), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
            mono = cv2.imread(str(paths["mono"]), cv2.IMREAD_GRAYSCALE)
            if semantic is None or depth is None or mono is None:
                missing.append(f"unreadable frame {pose['global_index']}")
                continue
            for raw, fields in labels.items():
                name = str(fields.get("class", "unlabelled")).lower()
                class_pixels[name] += int((semantic == int(raw)).sum())
            gate_ids = [
                int(raw)
                for raw, fields in labels.items()
                if str(fields.get("class", "")).lower() == "gate"
            ]
            gate_mask = np.isin(semantic, gate_ids)
            visible = bool(gate_mask.any())
            gate_visible += int(visible)
            if int(gate_mask.sum()) >= 50:
                gate_values = mono[gate_mask]
                gate_texture_bright_fractions.append(
                    float((gate_values > 100).mean())
                )
                gate_texture_contrasts.append(
                    float(
                        np.percentile(gate_values, 90)
                        - np.percentile(gate_values, 10)
                    )
                )
            gate_opening += int(valid_corner[local])
            gate_partial_or_occluded += int(visible and not valid_corner[local])
            invalid = depth == 0
            invalid_depth += int(invalid.sum())
            depth_pixels += depth.size
            all_invalid_depth += int(invalid.all())
            valid_values = depth[~invalid]
            if len(valid_values):
                depth_min_mm = min(depth_min_mm, int(valid_values.min()))
                depth_max_mm = max(depth_max_mm, int(valid_values.max()))
            frames += 1
            global_indices.append(int(pose["global_index"]))

    duplicates = {
        "exact_mono_duplicate_frames": sum(v - 1 for v in image_hashes.values() if v > 1),
        "duplicate_camera_poses": sum(v - 1 for v in pose_hashes.values() if v > 1),
    }
    texture_median_bright_fraction = (
        float(np.median(gate_texture_bright_fractions))
        if gate_texture_bright_fractions else 0.0
    )
    texture_median_contrast = (
        float(np.median(gate_texture_contrasts))
        if gate_texture_contrasts
        else 0.0
    )
    # Calibrated against the rejected v1 render and the corrected v2 render
    # at the actual 160x160 HM01B0 output resolution.  On an evenly sampled
    # 5,000-frame audit, v1 had a median bright fraction of 0.107 while v2
    # had 0.249.  Requiring both enough bright fabric/logo pixels and useful
    # within-gate contrast rejects the old nearly-black render without
    # coupling this guard to the exposure of a single smoke-test scene.
    texture_min_bright_fraction = 0.20
    texture_min_contrast_gray = 40.0
    texture_passed = (
        len(gate_texture_bright_fractions) >= 1000
        and texture_median_bright_fraction >= texture_min_bright_fraction
        and texture_median_contrast >= texture_min_contrast_gray
    )
    report = {
        "passed_structural_checks": not missing
        and not target_schema_errors
        and not scene_metadata_errors
        and texture_passed
        and frames == 50000
        and len(set(global_indices)) == 50000,
        "frames": frames,
        "missing_or_unreadable": missing[:100],
        "target_schema_errors": target_schema_errors[:100],
        "scene_metadata_errors": scene_metadata_errors[:100],
        "unique_global_indices": len(set(global_indices)),
        "duplicates": duplicates,
        "depth": {
            "encoding": "uint16 millimeters; zero invalid",
            "invalid_pixel_fraction": invalid_depth / depth_pixels,
            "all_invalid_frames": all_invalid_depth,
            "minimum_valid_mm": depth_min_mm,
            "maximum_valid_mm": depth_max_mm,
        },
        "gates": {
            "visible_frames": gate_visible,
            "closed_opening_and_ordered_corners": gate_opening,
            "visible_but_partial_or_occluded": gate_partial_or_occluded,
            "outside_frame_or_fully_occluded": frames - gate_visible,
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "texture_validation": {
                "version": "newbeedrone_hm01b0_v2",
                "frames_with_at_least_50_gate_pixels": len(
                    gate_texture_bright_fractions
                ),
                "median_fraction_above_gray_100": texture_median_bright_fraction,
                "minimum_median_fraction_above_gray_100": (
                    texture_min_bright_fraction
                ),
                "median_p90_minus_p10_gray": texture_median_contrast,
                "minimum_median_p90_minus_p10_gray": (
                    texture_min_contrast_gray
                ),
                "passed": texture_passed,
                "rationale": (
                    "thresholds calibrated at final 160x160 HM01B0 resolution; "
                    "rejects metadata-only or nearly solid-black v1 gate renders"
                ),
            },
        },
        "semantic_pixel_counts": dict(class_pixels),
        "distinct_label_maps": len(label_hashes),
        "scene_count": 1,
        "trajectory_count": 0,
        "timestamps": "absent",
        "vehicle_states": "absent",
        "simulator_seed_grouping": "50 shards; derived base seed plus shard start index",
        "caveat": "Exact duplicate hashing does not detect near-duplicate views.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed_structural_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
