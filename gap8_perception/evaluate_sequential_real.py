#!/usr/bin/env python3
"""Gate-corner evaluation of the canonical sequential model on real flights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .audit_real_flights import canonical_image_order
from .evaluate_real_flights import summarize
from .model_sequential import SequentialSTDCNet
from .quantization import prepare_int8_qat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flights", default="flight_08")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--label-tolerance-px", type=float, default=3.25)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SequentialSTDCNet()
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model = model.to(args.device).eval()
    model.load_state_dict(state["model"])
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "label_contract": state.get("label_contract"),
        "input_crop_sensor_rows": [20, 140],
        "dataset_scope": "gate-only real-flight evaluation; no clearance claim",
        "flights": {},
    }
    all_errors, all_detected = [], []
    excluded = 0
    with torch.no_grad():
        for flight in (item for item in args.flights.split(",") if item):
            folder = args.real_root / flight
            rows = [json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines() if line]
            errors, detected = [], []
            for start in range(0, len(rows), args.batch_size):
                images, truths = [], []
                for row in rows[start:start + args.batch_size]:
                    truth = canonical_image_order(np.asarray(row["corners"], np.float32).reshape(4, 2))[0]
                    if (truth[:, 1] < 20).any() or (truth[:, 1] >= 140).any():
                        excluded += 1
                        continue
                    image = cv2.imread(str(folder / "stream_out" / row["image"]), cv2.IMREAD_GRAYSCALE)
                    if image is None or image.shape != (160, 160):
                        raise ValueError(f"invalid real frame {row['image']}")
                    images.append(image[20:140])
                    truths.append(truth)
                if not images:
                    continue
                output = model(torch.from_numpy(np.asarray(images)).unsqueeze(1).float().to(args.device) / 255.0)
                logits = output[:, :4].cpu().numpy()
                predicted, scores = [], []
                for field in logits:
                    corners, corner_scores = [], []
                    for channel in field:
                        y, x = np.unravel_index(np.argmax(channel), channel.shape)
                        corners.append((8.0 * (x + 0.5) - 0.5, 8.0 * (y + 0.5) + 19.5))
                        corner_scores.append(float(channel.max()))
                    predicted.append(corners)
                    scores.append(corner_scores)
                values = np.linalg.norm(np.asarray(predicted) - np.asarray(truths), axis=2).mean(axis=1)
                errors.extend(values.tolist())
                detected.extend((np.asarray(scores) > 0.0).all(axis=1).tolist())
            values = np.asarray(errors, np.float32)
            found = np.asarray(detected, bool)
            report["flights"][flight] = summarize(values, found, args.label_tolerance_px)
            all_errors.append(values)
            all_detected.append(found)
    report["excluded_outside_crop"] = excluded
    report["aggregate"] = summarize(np.concatenate(all_errors), np.concatenate(all_detected), args.label_tolerance_px)
    report["clearance_metrics"] = None
    report["clearance_metrics_reason"] = "real flights lack obstacle/clearance ground truth"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
