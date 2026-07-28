#!/usr/bin/env python3
"""Reject integer exports that are too far from their trained float model."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-corner-peak-error", type=float, default=4.5)
    parser.add_argument("--max-danger-probability-mae", type=float, default=0.18)
    parser.add_argument("--min-images", type=int, default=32)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    parity = report["integer_parity"]
    failures = []
    if int(parity["images"]) < args.min_images:
        failures.append(
            "parity images %d < %d" % (parity["images"], args.min_images)
        )
    corner_error = float(parity["corner_peak_mean_error_heatmap_px"])
    if corner_error > args.max_corner_peak_error:
        failures.append(
            "corner peak mean error %.6f > %.6f"
            % (corner_error, args.max_corner_peak_error)
        )
    danger_error = float(parity["danger_probability_mae"])
    if danger_error > args.max_danger_probability_mae:
        failures.append(
            "danger probability MAE %.6f > %.6f"
            % (danger_error, args.max_danger_probability_mae)
        )
    if failures:
        raise SystemExit("integer parity gate failed: " + "; ".join(failures))
    print(
        "integer parity gate passed: images=%d corner_peak_mean=%.6f "
        "danger_probability_mae=%.6f"
        % (parity["images"], corner_error, danger_error)
    )


if __name__ == "__main__":
    main()
