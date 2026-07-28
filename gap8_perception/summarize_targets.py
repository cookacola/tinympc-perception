#!/usr/bin/env python3
"""Aggregate cached target summaries and split-level class balance."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split_file.read_text())
    report = {"splits": {}, "method": None}
    for name in ("train", "validation", "test"):
        frames = corner_valid = danger_positive = danger_cells = 0
        training_records = inverse_range_sum = inverse_range_cells = 0
        gate_positive = gate_pixels = 0
        global_indices = []
        by_speed: dict[str, dict[str, float]] = {}
        for shard in split[name]:
            archive = np.load(args.targets / f"{shard}.npz")
            frames += len(archive["corner_valid_u8"])
            corner_valid += int(archive["corner_valid_u8"].sum())
            danger_positive += int((archive["danger_u8"] >= 128).sum())
            danger_cells += archive["danger_u8"].size
            variants = archive["vehicle_state_f32"].shape[1]
            training_records += len(archive["corner_valid_u8"])
            inverse_range_sum += int(archive["inverse_range_u8"].sum())
            inverse_range_cells += archive["inverse_range_u8"].size
            for variant, speed in enumerate(archive["speed_variants_mps"]):
                key = f"{float(speed):g}"
                entry = by_speed.setdefault(
                    key, {"positive": 0.0, "cells": 0.0, "urgency": 0.0}
                )
                danger_variant = archive["danger_u8"][:, variant]
                urgency_variant = archive["urgency_u8"][:, variant]
                entry["positive"] += float((danger_variant >= 128).sum())
                entry["cells"] += float(danger_variant.size)
                entry["urgency"] += float(urgency_variant.sum())
            gate_positive += int(archive["gate_opening_u8"].sum())
            gate_pixels += archive["gate_opening_u8"].size
            global_indices.extend(archive["global_indices_i32"].tolist())
        report["splits"][name] = {
            "frames": frames,
            "training_records": training_records,
            "corner_valid_frames": corner_valid,
            "corner_valid_fraction": corner_valid / frames,
            "danger_positive_fraction": danger_positive / danger_cells,
            "mean_inverse_range": (
                inverse_range_sum / (255.0 * inverse_range_cells)
            ),
            "by_speed_mps": {
                speed: {
                    "danger_positive_fraction": values["positive"] / values["cells"],
                    "mean_urgency": values["urgency"] / (255.0 * values["cells"]),
                }
                for speed, values in sorted(by_speed.items(), key=lambda item: float(item[0]))
            },
            "gate_opening_positive_fraction": gate_positive / gate_pixels,
            "unique_global_indices": len(set(global_indices)),
        }
    summaries = sorted(args.targets.glob("shard_*.json"))
    if summaries:
        report["method"] = json.loads(summaries[0].read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
