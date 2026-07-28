#!/usr/bin/env python3
"""Evaluate corner, danger, gate, and deployment metrics on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from gap8_perception.data import MultiTaskDataset
from gap8_perception.danger_postprocessor import collision_probability_from_range
from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.quantization import prepare_int8_qat
from gap8_perception.gate_geometry import rotation_error_degrees
from gap8_perception.rollout_targets import load_calibration


def local_centroid(heatmaps: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Peak-local weighted centroid in heatmap coordinates."""
    batch, channels, height, width = heatmaps.shape
    output = heatmaps.new_zeros(batch, channels, 2)
    for b in range(batch):
        for c in range(channels):
            plane = heatmaps[b, c]
            peak = int(plane.argmax())
            py, px = divmod(peak, width)
            x0, x1 = max(0, px - radius), min(width, px + radius + 1)
            y0, y1 = max(0, py - radius), min(height, py + radius + 1)
            patch = plane[y0:y1, x0:x1].clamp_min(1e-8)
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=plane.device),
                torch.arange(x0, x1, device=plane.device),
                indexing="ij",
            )
            output[b, c, 0] = (patch * xx).sum() / patch.sum()
            output[b, c, 1] = (patch * yy).sum() / patch.sum()
    return output


def binary_counts(prediction, target):
    if isinstance(prediction, np.ndarray):
        prediction, target = prediction.astype(bool), target.astype(bool)
    else:
        prediction, target = prediction.bool(), target.bool()
    return {
        "tp": int((prediction & target).sum()),
        "fp": int((prediction & ~target).sum()),
        "fn": int((~prediction & target).sum()),
        "tn": int((~prediction & ~target).sum()),
    }


def safe_div(a, b):
    return float(a / b) if b else 0.0


def deployment_summary(model):
    macs, activations = [], []

    def hook(module, inputs, output):
        _, cout, height, width = output.shape
        kh, kw = module.kernel_size
        macs.append(cout * height * width * (module.in_channels // module.groups) * kh * kw)
        activations.append(output.numel())

    def linear_hook(module, inputs, output):
        macs.append(module.in_features * module.out_features)
        activations.append(output.numel())

    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    model(torch.zeros(1, 1, 160, 160, device=next(model.parameters()).device))
    for handle in handles:
        handle.remove()
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "conv_and_linear_macs": sum(macs),
        "peak_single_tensor_activation_elements": max(activations),
        "int8_parameter_bytes_estimate": sum(p.numel() for p in model.parameters()),
        "operators": [
            "Conv2d 3x3",
            "depthwise Conv2d 3x3",
            "pointwise Conv2d 1x1",
            "BatchNorm2d (fold at export)",
            "ReLU",
            "residual addition",
            "controller-side range-to-collision postprocessing (outside CNN)",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--evaluation-calibration",
        type=Path,
        default=Path("gap8_perception/configs/isaac_render_calibration.json"),
        help="Camera model used to render the evaluated synthetic images.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Gap8MultiTaskNet(
        state.get("gate_head", True), state.get("state_dim", 8)
    )
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = MultiTaskDataset(
        args.dataset, args.targets, args.split_file, args.split,
        expand_state_variants=True,
    )
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.workers
    )
    errors, all_four = [], 0
    valid_truth_frames = detected_truth_frames = 0
    invalid_truth_frames = false_gate_frames = 0
    danger_counts = dict(tp=0, fp=0, fn=0, tn=0)
    danger_thresholds = (
        0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005,
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04,
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    )
    danger_sweep_counts = {
        threshold: dict(tp=0, fp=0, fn=0, tn=0)
        for threshold in danger_thresholds
    }
    immediate_counts = dict(tp=0, fp=0, fn=0, tn=0)
    gate_counts = dict(tp=0, fp=0, fn=0, tn=0)
    gate_component_valid, gate_nonempty = 0, 0
    gate_quad_intersection = gate_quad_union = 0
    urgency_absolute_error = uncertainty_absolute_error = 0.0
    urgency_items = 0
    speed_records = {}
    danger_counts_by_speed = {}
    pnp_translation_errors = []
    pnp_rotation_errors = []
    pnp_attempts = pnp_successes = 0
    gate_projection_errors = []
    evaluation_K, evaluation_distortion = load_calibration(
        args.evaluation_calibration
    )
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            state_input = batch["vehicle_state"].to(device)
            outputs = model(image, state_input)
            urgency_prediction = outputs["urgency"].sigmoid().cpu().numpy()
            urgency_truth = batch["urgency"].numpy()
            uncertainty_prediction = outputs["uncertainty"].sigmoid().cpu().numpy()
            uncertainty_truth = batch["uncertainty"].numpy()
            urgency_absolute_error += float(
                np.abs(urgency_prediction - urgency_truth).sum()
            )
            uncertainty_absolute_error += float(
                np.abs(uncertainty_prediction - uncertainty_truth).sum()
            )
            urgency_items += urgency_truth.size
            base_hazard = outputs["danger"].sigmoid().cpu().numpy()
            danger_probability_float = np.empty_like(base_hazard)
            for frame in range(len(base_hazard)):
                state_values = batch["vehicle_state"][frame]
                danger_probability_float[frame], _ = collision_probability_from_range(
                    urgency_prediction[frame],
                    base_hazard[frame],
                    uncertainty_prediction[frame],
                    body_speed_mps=float(state_values[0]),
                    horizon_s=float(state_values[6]),
                    latency_s=float(state_values[7]),
                )
            for frame, global_index in enumerate(batch["global_index"].numpy()):
                speed = float(batch["vehicle_state"][frame, 0])
                speed_records.setdefault(int(global_index), {})[speed] = (
                    float(urgency_prediction[frame].mean()),
                    float(danger_probability_float[frame].mean()),
                )
            valid = batch["corner_valid"].numpy().astype(bool)
            corner_probability = outputs["corners"].sigmoid()
            confidence = corner_probability.flatten(2).amax(2).cpu().numpy()
            detected = (confidence >= 0.20).all(axis=1)
            pred_xy = local_centroid(corner_probability).cpu().numpy() * 4.0
            truth_xy = batch["corner_xy"].numpy()
            if valid.any():
                valid_truth_frames += int(valid.sum())
                detected_truth_frames += int((detected & valid).sum())
                frame_errors = np.linalg.norm(pred_xy[valid] - truth_xy[valid], axis=2)
                errors.extend(frame_errors.ravel().tolist())
                all_four += int((detected & valid).sum())
            for frame in range(len(valid)):
                # Geometry repeats across state variants; evaluate it once per
                # source image using authoritative fixed-course gate poses.
                if (
                    not valid[frame]
                    or not detected[frame]
                    or int(batch["state_variant"][frame]) != 0
                ):
                    continue
                gate_projection_errors.append(
                    float(batch["gate_projection_error_px"][frame])
                )
                pnp_attempts += 1
                success, rotation_vector, translation = cv2.solvePnP(
                    batch["gate_object_corners_m"][frame].numpy().astype(
                        np.float64
                    ),
                    pred_xy[frame].astype(np.float64),
                    evaluation_K.astype(np.float64),
                    evaluation_distortion.astype(np.float64),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    pnp_successes += 1
                    predicted_rotation = cv2.Rodrigues(rotation_vector)[0]
                    pnp_translation_errors.append(
                        float(
                            np.linalg.norm(
                                translation[:, 0]
                                - batch["gate_translation_camera_m"][
                                    frame
                                ].numpy()
                            )
                        )
                    )
                    pnp_rotation_errors.append(
                        rotation_error_degrees(
                            predicted_rotation,
                            batch["gate_rotation_camera"][frame].numpy(),
                        )
                    )
            invalid_truth_frames += int((~valid).sum())
            false_gate_frames += int((detected & ~valid).sum())

            danger_probability = danger_probability_float >= 0.5
            danger_truth = batch["danger"].numpy() >= 0.5
            for threshold, store in danger_sweep_counts.items():
                counts = binary_counts(
                    danger_probability_float >= threshold, danger_truth
                )
                for key in store:
                    store[key] += counts[key]
            ttc_truth = batch["time_to_collision_s"].numpy()
            immediate_truth = (ttc_truth >= 0.0) & (ttc_truth <= 0.30)
            for store, truth in (
                (danger_counts, danger_truth),
                (immediate_counts, immediate_truth),
            ):
                counts = binary_counts(danger_probability, truth)
                for key in store:
                    store[key] += counts[key]
            for frame in range(len(danger_truth)):
                speed = f"{float(batch['vehicle_state'][frame, 0]):g}"
                store = danger_counts_by_speed.setdefault(
                    speed, dict(tp=0, fp=0, fn=0, tn=0)
                )
                counts = binary_counts(
                    danger_probability[frame], danger_truth[frame]
                )
                for key in store:
                    store[key] += counts[key]

            if "gate" in outputs:
                gate_prediction = outputs["gate"].sigmoid().cpu().numpy() >= 0.5
                gate_truth = batch["gate"].numpy() >= 0.5
                counts = binary_counts(gate_prediction, gate_truth)
                for key in gate_counts:
                    gate_counts[key] += counts[key]
                for mask in gate_prediction[:, 0]:
                    if mask.any():
                        gate_nonempty += 1
                        components = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
                        gate_component_valid += int(components == 1)
                for frame, mask in enumerate(gate_prediction[:, 0]):
                    if not detected[frame]:
                        continue
                    quadrilateral = np.zeros((40, 40), np.uint8)
                    cv2.fillConvexPoly(
                        quadrilateral,
                        np.rint(pred_xy[frame] / 4.0).astype(np.int32),
                        1,
                    )
                    gate_quad_intersection += int(
                        ((quadrilateral > 0) & mask).sum()
                    )
                    gate_quad_union += int(
                        ((quadrilateral > 0) | mask).sum()
                    )

    errors = np.asarray(errors)
    danger = {
        "precision": safe_div(danger_counts["tp"], danger_counts["tp"] + danger_counts["fp"]),
        "recall": safe_div(danger_counts["tp"], danger_counts["tp"] + danger_counts["fn"]),
        "iou": safe_div(danger_counts["tp"], danger_counts["tp"] + danger_counts["fp"] + danger_counts["fn"]),
        "false_safe_rate": safe_div(danger_counts["fn"], danger_counts["tp"] + danger_counts["fn"]),
        "immediate_danger_recall": safe_div(immediate_counts["tp"], immediate_counts["tp"] + immediate_counts["fn"]),
        "counts": danger_counts,
        "by_speed_mps": {
            speed: {
                "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
                "false_safe_rate": safe_div(
                    counts["fn"], counts["tp"] + counts["fn"]
                ),
                "counts": counts,
            }
            for speed, counts in sorted(
                danger_counts_by_speed.items(), key=lambda item: float(item[0])
            )
        },
    }
    threshold_sweep = []
    for threshold, counts in danger_sweep_counts.items():
        threshold_sweep.append({
            "threshold": threshold,
            "precision": safe_div(
                counts["tp"], counts["tp"] + counts["fp"]
            ),
            "recall": safe_div(
                counts["tp"], counts["tp"] + counts["fn"]
            ),
            "false_safe_rate": safe_div(
                counts["fn"], counts["tp"] + counts["fn"]
            ),
            "counts": counts,
        })
    recall_95_candidates = [
        item for item in threshold_sweep if item["recall"] >= 0.95
    ]
    danger["threshold_sweep"] = threshold_sweep
    danger["recommended_threshold_for_recall_0.95"] = (
        max(recall_95_candidates, key=lambda item: item["precision"])
        if recall_95_candidates else None
    )
    danger["threshold_recommendation_authoritative"] = (
        args.split == "validation"
    )
    gate = {
        "iou": safe_div(gate_counts["tp"], gate_counts["tp"] + gate_counts["fp"] + gate_counts["fn"]),
        "dice": safe_div(2 * gate_counts["tp"], 2 * gate_counts["tp"] + gate_counts["fp"] + gate_counts["fn"]),
        "single_component_fraction_nonempty": safe_div(gate_component_valid, gate_nonempty),
        "corner_quadrilateral_overlap_iou": safe_div(
            gate_quad_intersection, gate_quad_union
        ),
        "counts": gate_counts,
    }
    report = {
        "split": args.split,
        "frames": len(dataset),
        "corner": {
            "valid_frames": int(len(errors) // 4),
            "mean_heatmap_px": float(errors.mean() / 4) if len(errors) else None,
            "mean_image_px": float(errors.mean()) if len(errors) else None,
            "median_image_px": float(np.median(errors)) if len(errors) else None,
            "p95_image_px": float(np.percentile(errors, 95)) if len(errors) else None,
            "pck_4px": float((errors <= 4).mean()) if len(errors) else None,
            "pck_8px": float((errors <= 8).mean()) if len(errors) else None,
            "pck_12px": float((errors <= 12).mean()) if len(errors) else None,
            "all_four_valid_frames": all_four,
            "all_four_detection_rate_on_valid": safe_div(
                detected_truth_frames, valid_truth_frames
            ),
            "false_gate_rate_on_invalid": safe_div(
                false_gate_frames, invalid_truth_frames
            ),
            "corner_peak_threshold": 0.20,
            "authoritative_gate_projection_mean_error_px": (
                float(np.mean(gate_projection_errors))
                if gate_projection_errors else None
            ),
            "pnp": {
                "attempts": pnp_attempts,
                "successes": pnp_successes,
                "success_rate": safe_div(pnp_successes, pnp_attempts),
                "translation_mean_error_m": (
                    float(np.mean(pnp_translation_errors))
                    if pnp_translation_errors else None
                ),
                "translation_p95_error_m": (
                    float(np.percentile(pnp_translation_errors, 95))
                    if pnp_translation_errors else None
                ),
                "rotation_mean_error_deg": (
                    float(np.mean(pnp_rotation_errors))
                    if pnp_rotation_errors else None
                ),
                "rotation_p95_error_deg": (
                    float(np.percentile(pnp_rotation_errors, 95))
                    if pnp_rotation_errors else None
                ),
                "evaluation_camera_model": str(args.evaluation_calibration),
            },
        },
        "danger": danger,
        "urgency": {
            "mean_absolute_error": urgency_absolute_error / urgency_items,
            "uncertainty_mean_absolute_error": (
                uncertainty_absolute_error / urgency_items
            ),
            "inverse_range_invariance_fraction_high_eq_low": safe_div(
                sum(
                    abs(values[max(values)][0] - values[min(values)][0]) < 1e-6
                    for values in speed_records.values()
                    if len(values) >= 2
                ),
                sum(len(values) >= 2 for values in speed_records.values()),
            ),
            "danger_sensitivity_fraction_high_gt_low": safe_div(
                sum(
                    values[max(values)][1] > values[min(values)][1]
                    for values in speed_records.values()
                    if len(values) >= 2
                ),
                sum(len(values) >= 2 for values in speed_records.values()),
            ),
        },
        "gate": gate,
        "deployment": deployment_summary(model),
        "unavailable_metrics": [
            "closed-loop safe-corridor collision accuracy",
            "closed-loop collision rate",
            "measured GAP8 latency and L1/L2 tiling",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
