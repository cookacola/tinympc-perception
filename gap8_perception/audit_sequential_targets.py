#!/usr/bin/env python3
"""Validate the geometry-label corpus required by SequentialSTDCNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .output_contract import OFFSET_MAX, OFFSET_MIN
except ImportError:  # Support direct Slurm/script invocation as well as -m.
    from output_contract import OFFSET_MAX, OFFSET_MIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archives = sorted(args.targets.glob("shard_*.npz"))
    if len(archives) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} target shards, found {len(archives)}")
    required = {
        "schema", "fixed_normal_offsets_m_f16", "fixed_normal_confidence_u8",
        "fixed_normal_angles_deg_f32", "corners_xy160_f16",
        "corner_visibility_u8", "corner_valid_u8", "corner_label_convention",
    }
    forbidden = {
        "danger_u8", "urgency_u8", "inverse_range_u8", "uncertainty_u8",
        "vehicle_state_f32", "speed_variants_mps",
    }
    rows = 0
    offsets = []
    confidence = []
    for archive_path in archives:
        with np.load(archive_path) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"{archive_path}: missing {sorted(missing)}")
            unexpected = forbidden.intersection(archive.files)
            if unexpected:
                raise RuntimeError(f"{archive_path}: legacy targets present {sorted(unexpected)}")
            if str(archive["schema"].item()) != "sequential_fixed_normal_v1":
                raise RuntimeError(f"{archive_path}: wrong target schema")
            if str(archive["corner_label_convention"].item()) != "gate_frame_centerline":
                raise RuntimeError(f"{archive_path}: wrong corner convention")
            current_offsets = archive["fixed_normal_offsets_m_f16"].astype(np.float32)
            current_confidence = archive["fixed_normal_confidence_u8"]
            if current_offsets.ndim != 2 or current_offsets.shape[1] != 4:
                raise RuntimeError(f"{archive_path}: invalid offset shape {current_offsets.shape}")
            if current_confidence.shape != current_offsets.shape:
                raise RuntimeError(f"{archive_path}: confidence shape mismatch")
            if not np.isfinite(current_offsets).all():
                raise RuntimeError(f"{archive_path}: non-finite offsets")
            if (current_offsets < OFFSET_MIN - 1e-3).any() or (current_offsets > OFFSET_MAX + 1e-3).any():
                raise RuntimeError(f"{archive_path}: offsets outside [{OFFSET_MIN}, {OFFSET_MAX}]")
            if not np.isin(current_confidence, (0, 255)).all():
                raise RuntimeError(f"{archive_path}: confidence is not binary uint8")
            rows += len(current_offsets)
            offsets.append(current_offsets)
            confidence.append(current_confidence)
    offsets = np.concatenate(offsets)
    confidence = np.concatenate(confidence)
    report = {
        "passed": True,
        "target_root": str(args.targets),
        "shards": len(archives),
        "records": rows,
        "offset_range_m": [float(offsets.min()), float(offsets.max())],
        "offset_mean_m": offsets.mean(axis=0).tolist(),
        "confidence_visible_fraction": (confidence.mean(axis=0) / 255.0).tolist(),
    }
    if any(value < 0.95 for value in report["confidence_visible_fraction"]):
        raise RuntimeError(
            "one or more fixed normals lack image support; recalibrate normal angles"
        )
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
