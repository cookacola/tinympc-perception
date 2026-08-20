#!/usr/bin/env python3
"""Compose the calibrated navigation and exact integer gate release report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integer-evaluation", type=Path, required=True)
    parser.add_argument("--navigation-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.integer_evaluation.read_text())
    calibration = json.loads(args.navigation_calibration.read_text())
    result = dict(original)
    result["format"] = "espnet-dronet-middle-gate-release-evaluation-v1"
    result["navigation_validation"] = calibration["validation"]
    result["navigation_test"] = calibration["test"]
    result["navigation_output_calibration"] = calibration
    result["navigation_threshold_selected_on"] = "official validation split"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
