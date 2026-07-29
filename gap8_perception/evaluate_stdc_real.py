#!/usr/bin/env python3
"""Gate-only real-flight evaluation for the 160x120 STDC student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.evaluate import local_centroid
from gap8_perception.evaluate_real_flights import summarize
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet, ProposedSTDCFPNNet
from gap8_perception.model_stdc_dory import Gap8STDCCornerDoryNet
from gap8_perception.model_stdc_dory import Gap8STDCSharedDoryNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--real-root", type=Path, default=Path("/home/cchen/real_flight_data")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--label-tolerance-px", type=float, default=3.25)
    parser.add_argument(
        "--flights",
        default="flight_06,flight_07,flight_08",
        help="Comma-separated whole-flight evaluation split.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dory_pair = state.get("architecture") == "Gap8STDCDoryPair"
    shared_dory = state.get("architecture") == "Gap8STDCSharedDoryNet"
    model = (
        Gap8STDCCornerDoryNet()
        if dory_pair
        else Gap8STDCSharedDoryNet()
        if shared_dory
        else ProposedSTDCFPNNet()
        if state.get("architecture") == "ProposedSTDCFPNNet"
        else Gap8STDCMultiHeadNet()
    ).to(device)
    model.load_state_dict(
        state["corner_model"] if dory_pair else state["model"]
    )
    model.eval()
    report = {
        "checkpoint": str(args.checkpoint),
        "input_crop_sensor_rows": [20, 140],
        "dataset_scope": "gate-only; no real danger claim",
        "flights": {},
    }
    all_errors, all_detected = [], []
    excluded = 0
    with torch.no_grad():
        for flight in tuple(item for item in args.flights.split(",") if item):
            folder = args.real_root / flight
            rows = [
                json.loads(line)
                for line in (folder / "labels.jsonl").read_text().splitlines()
                if line
            ]
            flight_errors, flight_detected = [], []
            for start in range(0, len(rows), args.batch_size):
                images, truths = [], []
                for row in rows[start : start + args.batch_size]:
                    image = cv2.imread(
                        str(folder / "stream_out" / row["image"]),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    truth = canonical_image_order(
                        np.asarray(row["corners"], np.float32).reshape(4, 2)
                    )[0]
                    if (truth[:, 1] < 20).any() or (truth[:, 1] >= 140).any():
                        excluded += 1
                        continue
                    images.append(image[20:140])
                    truths.append(truth)
                if not images:
                    continue
                tensor = (
                    torch.from_numpy(np.asarray(images))
                    .unsqueeze(1)
                    .float()
                    .to(device)
                    / 255.0
                )
                if dory_pair or shared_dory:
                    logits = model(tensor)
                    corner_probability = (
                        logits if dory_pair else logits["corners"]
                    ).sigmoid()
                    confidence = (
                        corner_probability.flatten(2).amax(2).cpu().numpy()
                    )
                else:
                    outputs = model.predict(tensor)
                    corner_probability = outputs["corners"]
                    confidence = outputs["corner_confidence"].cpu().numpy()
                prediction = local_centroid(corner_probability).cpu().numpy()
                prediction[..., 0] *= 4.0
                prediction[..., 1] = prediction[..., 1] * 4.0 + 20.0
                detected = (confidence >= 0.25).all(axis=1)
                errors = np.linalg.norm(
                    prediction - np.asarray(truths), axis=2
                ).mean(axis=1)
                flight_errors.extend(errors.tolist())
                flight_detected.extend(detected.tolist())
            values = np.asarray(flight_errors, np.float32)
            detections = np.asarray(flight_detected, bool)
            report["flights"][flight] = summarize(
                values, detections, args.label_tolerance_px
            )
            all_errors.append(values)
            all_detected.append(detections)
    report["excluded_outside_crop"] = excluded
    report["aggregate"] = summarize(
        np.concatenate(all_errors),
        np.concatenate(all_detected),
        args.label_tolerance_px,
    )
    report["danger_metrics"] = None
    report["danger_metrics_reason"] = (
        "real flights contain gates but no obstacles or collision labels"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
