#!/usr/bin/env python3
"""Verify rendered semantic gates agree with configured K/distortion and world poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gap8_perception.gate_geometry import associate_gate
from gap8_perception.rollout_targets import load_calibration
from gap8_perception.targets import class_mask, gate_opening_and_corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-mean-error-px", type=float, default=3.0)
    args = parser.parse_args()
    K, distortion = load_calibration(args.calibration)
    poses = [
        json.loads(line)
        for line in (args.shard / "poses.jsonl").read_text().splitlines()
    ]
    errors = []
    for pose in poses:
        local = pose["local_index"]
        suffix = f"{local:04d}"
        gate = class_mask(
            args.shard / f"semantic_segmentation_{suffix}.png",
            args.shard / f"semantic_segmentation_labels_{suffix}.json",
            {"gate"},
        )
        _, corners, valid = gate_opening_and_corners(gate)
        if not valid:
            continue
        _, _, error = associate_gate(
            corners, pose["eye_m"], pose["target_m"], K, distortion
        )
        errors.append(error)
    if not errors:
        raise RuntimeError("no valid gate openings available for calibration check")
    report = {
        "shard": str(args.shard),
        "valid_gate_frames": len(errors),
        "mean_projection_error_px": float(np.mean(errors)),
        "median_projection_error_px": float(np.median(errors)),
        "p95_projection_error_px": float(np.percentile(errors, 95)),
        "maximum_allowed_mean_error_px": args.maximum_mean_error_px,
        "passed": float(np.mean(errors)) <= args.maximum_mean_error_px,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("rendered camera does not match supplied calibration")


if __name__ == "__main__":
    main()
