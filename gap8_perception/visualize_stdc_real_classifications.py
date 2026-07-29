#!/usr/bin/env python3
"""Overlay STDC gate/danger classifications and audit systematic label bias."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.evaluate import local_centroid
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet
from gap8_perception.model_stdc_dory import Gap8STDCSharedDoryNet
from gap8_perception.postprocess_stdc import (
    GateDecision,
    gate_override_danger,
    validate_gate_geometry,
)


COLORS = ((255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80))
NAMES = ("TL", "TR", "BR", "BL")


def bias_summary(prediction: np.ndarray, truth: np.ndarray) -> dict:
    residual = prediction - truth
    raw = np.linalg.norm(residual, axis=2).mean(axis=1)
    mean_offset = residual.reshape(-1, 2).mean(axis=0)
    translated = np.linalg.norm(residual - mean_offset, axis=2).mean(axis=1)
    design = np.concatenate(
        (truth.reshape(-1, 2), np.ones((truth.shape[0] * 4, 1))), axis=1
    )
    affine, _, _, _ = np.linalg.lstsq(design, prediction.reshape(-1, 2), rcond=None)
    affine_prediction = (design @ affine).reshape(-1, 4, 2)
    affine_error = np.linalg.norm(affine_prediction - prediction, axis=2).mean(axis=1)
    truth_center = truth.mean(axis=1)
    pred_center = prediction.mean(axis=1)
    truth_width = (
        np.linalg.norm(truth[:, 1] - truth[:, 0], axis=1)
        + np.linalg.norm(truth[:, 2] - truth[:, 3], axis=1)
    ) / 2.0
    pred_width = (
        np.linalg.norm(prediction[:, 1] - prediction[:, 0], axis=1)
        + np.linalg.norm(prediction[:, 2] - prediction[:, 3], axis=1)
    ) / 2.0
    truth_height = (
        np.linalg.norm(truth[:, 3] - truth[:, 0], axis=1)
        + np.linalg.norm(truth[:, 2] - truth[:, 1], axis=1)
    ) / 2.0
    pred_height = (
        np.linalg.norm(prediction[:, 3] - prediction[:, 0], axis=1)
        + np.linalg.norm(prediction[:, 2] - prediction[:, 1], axis=1)
    ) / 2.0
    return {
        "frames": int(len(prediction)),
        "raw_mean_frame_corner_error_px": float(raw.mean()),
        "raw_median_frame_corner_error_px": float(np.median(raw)),
        "mean_prediction_minus_label_xy_px": mean_offset.tolist(),
        "per_corner_prediction_minus_label_xy_px": residual.mean(axis=0).tolist(),
        "translation_corrected_mean_error_px": float(translated.mean()),
        "translation_explained_error_fraction": float(
            1.0 - translated.mean() / max(raw.mean(), 1e-9)
        ),
        "affine_label_to_prediction": affine.tolist(),
        "affine_fit_residual_to_prediction_px": float(affine_error.mean()),
        "center_prediction_minus_label_xy_px": (
            pred_center - truth_center
        ).mean(axis=0).tolist(),
        "median_predicted_to_label_width_ratio": float(
            np.median(pred_width / np.maximum(truth_width, 1e-6))
        ),
        "median_predicted_to_label_height_ratio": float(
            np.median(pred_height / np.maximum(truth_height, 1e-6))
        ),
    }


def render_overlay(record, output_path: Path):
    image = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left = base.copy()
    truth = record["truth"]
    prediction = record["prediction"]
    cv2.polylines(left, [np.rint(truth).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(
        left, [np.rint(prediction).astype(np.int32)], True, (0, 255, 0), 2
    )
    for name, color, source, target in zip(
        NAMES, COLORS, truth, prediction
    ):
        source_i = tuple(np.rint(source).astype(int))
        target_i = tuple(np.rint(target).astype(int))
        cv2.arrowedLine(left, source_i, target_i, color, 1, tipLength=0.2)
        cv2.putText(
            left,
            name,
            (target_i[0] + 2, target_i[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )
    danger = record["danger"]
    danger_color = cv2.applyColorMap(
        cv2.resize(
            np.uint8(np.clip(danger * 255.0, 0, 255)),
            (160, 120),
            interpolation=cv2.INTER_NEAREST,
        ),
        cv2.COLORMAP_JET,
    )
    right_crop = cv2.addWeighted(base[20:140], 0.45, danger_color, 0.55, 0)
    right = base.copy()
    right[20:140] = right_crop
    cv2.polylines(
        right, [np.rint(prediction).astype(np.int32)], True, (255, 255, 255), 1
    )
    status = "GATE ACCEPT" if record["accepted"] else "GATE REJECT"
    status_color = (0, 220, 0) if record["accepted"] else (0, 0, 255)
    for panel in (left, right):
        cv2.rectangle(panel, (0, 0), (159, 18), (0, 0, 0), -1)
    cv2.putText(
        left,
        f"{status} err={record['error']:.1f}px conf={record['min_conf']:.2f}",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        f"danger mean={danger.mean():.2f} max={danger.max():.2f}",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        left,
        "red=label green=model arrows=label->model",
        (2, 157),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.25,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        np.hstack((left, right)),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--real-root", type=Path, default=Path("/home/cchen/real_flight_data")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    shared_dory = state.get("architecture") == "Gap8STDCSharedDoryNet"
    model = (
        Gap8STDCSharedDoryNet() if shared_dory else Gap8STDCMultiHeadNet()
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    records = []
    with torch.no_grad():
        for flight in ("flight_06", "flight_07", "flight_08"):
            folder = args.real_root / flight
            rows = [
                json.loads(line)
                for line in (folder / "labels.jsonl").read_text().splitlines()
                if line
            ]
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start : start + args.batch_size]
                images, truths, paths = [], [], []
                for row in chunk:
                    path = folder / "stream_out" / row["image"]
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    images.append(image[20:140])
                    truths.append(
                        canonical_image_order(
                            np.asarray(row["corners"], np.float32).reshape(4, 2)
                        )[0]
                    )
                    paths.append(path)
                tensor = (
                    torch.from_numpy(np.asarray(images))
                    .unsqueeze(1)
                    .float()
                    .to(device)
                    / 255.0
                )
                if shared_dory:
                    logits = model(tensor)
                    corner_probability = logits["corners"].sigmoid()
                    confidence = (
                        corner_probability.flatten(2).amax(2).cpu().numpy()
                    )
                    danger = logits["danger"].sigmoid().cpu().numpy()[:, 0]
                else:
                    outputs = model.predict(tensor)
                    corner_probability = outputs["corners"]
                    confidence = outputs["corner_confidence"].cpu().numpy()
                    danger = outputs["danger"].cpu().numpy()[:, 0]
                prediction = local_centroid(corner_probability).cpu().numpy()
                prediction[..., 0] *= 4.0
                prediction[..., 1] = prediction[..., 1] * 4.0 + 20.0
                for local in range(len(chunk)):
                    decision = validate_gate_geometry(
                        prediction[local], confidence[local]
                    )
                    truth = truths[local]
                    records.append(
                        {
                            "flight": flight,
                            "path": str(paths[local]),
                            "stem": paths[local].stem,
                            "truth": truth,
                            "prediction": prediction[local],
                            "confidence": confidence[local],
                            "min_conf": float(confidence[local].min()),
                            "danger": danger[local],
                            "accepted": decision.accepted,
                            "reason": decision.reason,
                            "error": float(
                                np.linalg.norm(
                                    prediction[local] - truth, axis=1
                                ).mean()
                            ),
                            "outside_crop": bool(
                                (truth[:, 1] < 20).any()
                                or (truth[:, 1] >= 140).any()
                            ),
                        }
                    )
    prediction = np.asarray([record["prediction"] for record in records])
    truth = np.asarray([record["truth"] for record in records])
    report = {
        "checkpoint": str(args.checkpoint),
        "legend": {
            "red": "mocap-projected/canonicalized label",
            "green": "model quadrilateral",
            "arrows": "label-to-model residual",
            "right_panel": (
                "raw danger probability (blue low, red high); visualization "
                "only because these real frames have no obstacle labels"
            ),
        },
        "dataset_scope": "gate-only; no real danger accuracy claim",
        "all": bias_summary(prediction, truth),
        "in_crop": bias_summary(
            np.asarray(
                [
                    record["prediction"]
                    for record in records
                    if not record["outside_crop"]
                ]
            ),
            np.asarray(
                [
                    record["truth"]
                    for record in records
                    if not record["outside_crop"]
                ]
            ),
        ),
        "by_flight": {},
        "geometry_acceptance_rate": float(
            np.mean([record["accepted"] for record in records])
        ),
        "outside_crop_frames": int(sum(record["outside_crop"] for record in records)),
    }
    for flight in ("flight_06", "flight_07", "flight_08"):
        selected = [record for record in records if record["flight"] == flight]
        report["by_flight"][flight] = bias_summary(
            np.asarray([record["prediction"] for record in selected]),
            np.asarray([record["truth"] for record in selected]),
        )
        selected_in_crop = [
            record for record in selected if not record["outside_crop"]
        ]
        report["by_flight"][flight]["in_crop"] = bias_summary(
            np.asarray([record["prediction"] for record in selected_in_crop]),
            np.asarray([record["truth"] for record in selected_in_crop]),
        )
    (args.output / "bias_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "flight", "image", "mean_corner_error_px", "min_confidence",
                "gate_accepted", "rejection_reason", "outside_crop",
                "danger_mean", "danger_max",
            ]
            + [
                f"{prefix}_{name}_{axis}"
                for prefix in ("label", "prediction", "residual")
                for name in NAMES
                for axis in ("x", "y")
            ]
        )
        for record in records:
            residual = record["prediction"] - record["truth"]
            writer.writerow(
                [
                    record["flight"],
                    Path(record["path"]).name,
                    record["error"],
                    record["min_conf"],
                    int(record["accepted"]),
                    record["reason"],
                    int(record["outside_crop"]),
                    float(record["danger"].mean()),
                    float(record["danger"].max()),
                ]
                + record["truth"].ravel().tolist()
                + record["prediction"].ravel().tolist()
                + residual.ravel().tolist()
            )
    errors = np.asarray([record["error"] for record in records])
    selected_indices = set(range(0, len(records), max(args.stride, 1)))
    selected_indices.update(np.argsort(errors)[:100].tolist())
    selected_indices.update(np.argsort(errors)[-100:].tolist())
    for index in sorted(selected_indices):
        record = records[index]
        render_overlay(
            record,
            args.output
            / "overlays"
            / record["flight"]
            / f"{record['stem']}_overlay.jpg",
        )
    montage_records = [
        records[index]
        for index in np.linspace(0, len(records) - 1, 100).astype(int)
    ]
    montage_tiles = []
    temporary = args.output / ".montage_tiles"
    for index, record in enumerate(montage_records):
        path = temporary / f"{index:03d}.jpg"
        render_overlay(record, path)
        montage_tiles.append(cv2.resize(cv2.imread(str(path)), (320, 160)))
    montage = np.vstack(
        [np.hstack(montage_tiles[start : start + 5]) for start in range(0, 100, 5)]
    )
    cv2.imwrite(
        str(args.output / "classification_montage_100.jpg"),
        montage,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
