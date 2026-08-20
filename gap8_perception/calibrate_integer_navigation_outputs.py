#!/usr/bin/env python3
"""Fit validation-only affine yaw and Platt collision calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from gap8_perception.evaluate_espnet_dronet_gate_integer import (
    IntegerGraphs, MatchedFrames, best_f1_threshold, find_strict_pairs,
    navigation_raw, navigation_report,
)


def calibrated(raw, yaw_scale, yaw_bias, collision_scale, collision_bias):
    truth, scores, yaw_truth, yaws = raw
    logits = np.log(np.clip(scores, 1e-7, 1 - 1e-7) /
                    np.clip(1 - scores, 1e-7, 1 - 1e-7))
    yaws = yaws * yaw_scale + yaw_bias
    logits = logits * collision_scale + collision_bias
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return truth, scores, yaw_truth, yaws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integer-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--navigation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graphs = IntegerGraphs(args.integer_dir, args.report)
    valid = navigation_raw(
        graphs, MatchedFrames(find_strict_pairs(args.navigation_data, "valid")), 0
    )
    test = navigation_raw(
        graphs, MatchedFrames(find_strict_pairs(args.navigation_data, "test")), 0
    )
    yaw_model = LinearRegression().fit(valid[3].reshape(-1, 1), valid[2])
    valid_logits = np.log(np.clip(valid[1], 1e-7, 1 - 1e-7) /
                          np.clip(1 - valid[1], 1e-7, 1 - 1e-7))
    collision_model = LogisticRegression(C=1e6, max_iter=1000).fit(
        valid_logits.reshape(-1, 1), valid[0]
    )
    values = {
        "yaw_scale": float(yaw_model.coef_[0]),
        "yaw_bias": float(yaw_model.intercept_),
        "collision_logit_scale": float(collision_model.coef_[0, 0]),
        "collision_logit_bias": float(collision_model.intercept_[0]),
    }
    calibrated_valid = calibrated(valid, values["yaw_scale"], values["yaw_bias"],
                                  values["collision_logit_scale"],
                                  values["collision_logit_bias"])
    threshold = best_f1_threshold(calibrated_valid[0], calibrated_valid[1])
    report = {"format": "espnet-integer-navigation-calibration-v1",
              "selected_on": "official validation split", **values,
              "validation": navigation_report(calibrated_valid, threshold),
              "test": navigation_report(
                  calibrated(test, values["yaw_scale"], values["yaw_bias"],
                             values["collision_logit_scale"],
                             values["collision_logit_bias"]), threshold),
              "cnn_mac_overhead": 0, "scalar_postprocess_multiply_adds": 2}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
