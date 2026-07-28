#!/usr/bin/env python3
"""Held-out sim-to-real corner evaluation with noisy-label tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.evaluate import local_centroid
from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.quantization import prepare_int8_qat


def summarize(errors: np.ndarray, detected: np.ndarray, tolerance_px: float):
    adjusted = np.maximum(errors - tolerance_px, 0.0)
    return {
        "frames": int(len(errors)),
        "detection_rate": float(detected.mean()) if len(detected) else None,
        "raw_vs_noisy_label": {
            "mean_px": float(errors.mean()),
            "median_px": float(np.median(errors)),
            "p95_px": float(np.percentile(errors, 95)),
            "pck_4px": float((errors <= 4).mean()),
            "pck_8px": float((errors <= 8).mean()),
        },
        "tolerance_adjusted_upper_bound": {
            "label_tolerance_px": tolerance_px,
            "mean_excess_px": float(adjusted.mean()),
            "p95_excess_px": float(np.percentile(adjusted, 95)),
            "note": (
                "subtracts the independently audited label-to-edge tolerance; "
                "not a replacement for manually corrected ground truth"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--real-root", type=Path, default=Path("/home/cchen/real_flight_data")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualizations", type=Path)
    parser.add_argument("--label-tolerance-px", type=float, default=3.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Gap8MultiTaskNet(
        checkpoint.get("gate_head", True), checkpoint.get("state_dim", 8)
    )
    if checkpoint.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_scope": {
            "scene_content": "gates_only_no_obstacles",
            "valid_tasks": ["gate_corner_detection", "gate_geometry_verification"],
            "excluded_tasks": [
                "obstacle_presence",
                "inverse_range",
                "collision_danger",
                "collision_urgency",
            ],
            "unlabeled_frame_policy": (
                "missing gate labels are not obstacle or danger negatives"
            ),
        },
        "label_policy": (
            "per-flight labels canonicalized to image TL,TR,BR,BL; raw order "
            "is not trusted"
        ),
        "flights": {},
        "danger_metrics": None,
        "danger_metrics_reason": (
            "all real scenes contain gates only and no obstacles; the set has "
            "no obstacle geometry, collision labels, or counterfactual steering "
            "rollouts, so danger/range outputs are neither trained nor scored"
        ),
    }
    visualization_records = []
    all_errors = []
    all_detected = []
    colors = ((255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255))

    with torch.no_grad():
        for flight in ("flight_06", "flight_07", "flight_08"):
            folder = args.real_root / flight
            rows = [
                json.loads(line)
                for line in (folder / "labels.jsonl").read_text().splitlines()
                if line
            ]
            flight_errors = []
            flight_detected = []
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start : start + args.batch_size]
                images = []
                truths = []
                for row in chunk:
                    image = cv2.imread(
                        str(folder / "stream_out" / row["image"]),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    images.append(image)
                    raw = np.asarray(row["corners"], np.float32).reshape(4, 2)
                    truths.append(canonical_image_order(raw)[0])
                tensor = torch.from_numpy(np.asarray(images)).unsqueeze(1).float()
                tensor = tensor.to(device) / 255.0
                output = model.forward_image(tensor)
                probabilities = output["corners"].sigmoid()
                predicted = local_centroid(probabilities).cpu().numpy() * 4.0
                confidence = probabilities.flatten(2).amax(2).cpu().numpy()
                detected = (confidence >= 0.20).all(axis=1)
                errors = np.linalg.norm(
                    predicted - np.asarray(truths, np.float32), axis=2
                ).mean(axis=1)
                flight_errors.extend(errors.tolist())
                flight_detected.extend(detected.tolist())
                stride = max(1, len(rows) // 34)
                for local, row in enumerate(chunk):
                    absolute = start + local
                    if absolute % stride == 0:
                        visualization_records.append(
                            (
                                folder / "stream_out" / row["image"],
                                truths[local],
                                predicted[local],
                                confidence[local],
                                flight,
                            )
                        )
            flight_errors = np.asarray(flight_errors, np.float32)
            flight_detected = np.asarray(flight_detected, bool)
            report["flights"][flight] = summarize(
                flight_errors, flight_detected, args.label_tolerance_px
            )
            all_errors.append(flight_errors)
            all_detected.append(flight_detected)

    report["aggregate"] = summarize(
        np.concatenate(all_errors),
        np.concatenate(all_detected),
        args.label_tolerance_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.visualizations:
        args.visualizations.mkdir(parents=True, exist_ok=True)
        tiles = []
        for image_path, truth, prediction, confidence, flight in visualization_records[:100]:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(tile, [np.rint(truth).astype(np.int32)], True, (0, 0, 255), 1)
            for point, color, score in zip(prediction, colors, confidence):
                if score >= 0.20:
                    cv2.drawMarker(
                        tile, tuple(np.rint(point).astype(int)), color,
                        cv2.MARKER_CROSS, 7, 1,
                    )
            cv2.putText(
                tile, f"{flight}:{image_path.stem}", (2, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        rows = []
        for start in range(0, len(tiles), 10):
            row = tiles[start : start + 10]
            row.extend([np.zeros_like(tiles[0])] * (10 - len(row)))
            rows.append(np.hstack(row))
        cv2.imwrite(
            str(args.visualizations / "real_flight_predictions_100.png"),
            np.vstack(rows),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
