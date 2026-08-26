#!/usr/bin/env python3
"""Render representative natural-test gate heatmaps, corners, and visibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .ttc_gate_data import TTCGateDataset
from .ttc_gate_losses import peak_gate_coordinates
from .ttc_motion_gate_dory_model import load_dory_checkpoint


COLORS_BGR = ((255, 100, 40), (40, 220, 255), (255, 60, 220), (60, 255, 80))


def update(candidates, name, score, index, record, prefer_high):
    previous = candidates.get(name)
    if previous is None or (score > previous["score"] if prefer_high else score < previous["score"]):
        candidates[name] = {"score": float(score), "dataset_index": int(index), **record}


def scan(model, dataset, device, batch_size, workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    candidates, offset = {}, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            onboard = batch["onboard_state"].to(device, non_blocking=True)
            output = model(images, onboard)
            coordinates = peak_gate_coordinates(output["gate_heatmap_logits"])
            truth = batch["gate_corners_px"].to(device, non_blocking=True)
            visible = batch["gate_corners_visible"].to(device, non_blocking=True).bool()
            eligible = batch["gate_supervision_eligible"].bool()
            visibility_probability = output["gate_visibility_logits"].sigmoid()
            error = torch.linalg.vector_norm(coordinates - truth, dim=-1)
            for local in range(images.shape[0]):
                if not bool(eligible[local]):
                    continue
                count = int(visible[local].sum())
                mean_error = float(error[local][visible[local]].mean()) if count else 0.0
                record = {
                    "visible_count": count,
                    "mean_visible_error_px": mean_error,
                    "mean_visibility_probability": float(visibility_probability[local].mean()),
                    "gate_distance_m": float(batch["gate_distance_m"][local]),
                    "gate_projected_width_px": float(batch["gate_projected_width_px"][local]),
                    "gate_projected_height_px": float(batch["gate_projected_height_px"][local]),
                    "gate_projected_area_px2": float(batch["gate_projected_area_px2"][local]),
                }
                index = offset + local
                if count == 0:
                    update(candidates, "no_gate_correct", record["mean_visibility_probability"],
                           index, record, False)
                    update(candidates, "no_gate_false_alarm", record["mean_visibility_probability"],
                           index, record, True)
                elif count == 1:
                    update(candidates, "one_corner", mean_error, index, record, False)
                elif count == 2:
                    update(candidates, "two_corners", mean_error, index, record, False)
                elif count == 3:
                    update(candidates, "three_corners", mean_error, index, record, False)
                elif count == 4:
                    update(candidates, "full_gate_good", mean_error, index, record, False)
                    update(candidates, "full_gate_hard", min(mean_error, 80.0), index, record, True)
            offset += images.shape[0]
    return candidates


def label(panel, text):
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 19), (0, 0, 0), -1)
    cv2.putText(output, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render(model, dataset, name, candidate, device, output_dir):
    sample = dataset[candidate["dataset_index"]]
    with torch.no_grad():
        output = model(
            sample["images"][None].to(device), sample["onboard_state"][None].to(device)
        )
    coordinates = peak_gate_coordinates(output["gate_heatmap_logits"])[0].cpu().numpy()
    visibility = output["gate_visibility_logits"][0].sigmoid().cpu().numpy()
    heatmap = output["gate_heatmap_logits"][0].sigmoid().amax(0).cpu().numpy()
    image = np.rint(sample["images"][1].numpy() * 255.0).astype(np.uint8)
    mono = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    truth_panel = mono.copy()
    for point, valid, color in zip(
        sample["gate_corners_px"].numpy(), sample["gate_corners_visible"].numpy(), COLORS_BGR
    ):
        if valid:
            cv2.circle(truth_panel, tuple(np.rint(point).astype(int)), 4, color, -1)
    heatmap_panel = cv2.applyColorMap(
        cv2.resize(np.uint8(heatmap * 255), (160, 160), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_TURBO,
    )
    prediction_panel = cv2.addWeighted(mono, 0.52, heatmap_panel, 0.48, 0.0)
    for point, probability, color in zip(coordinates, visibility, COLORS_BGR):
        if probability >= 0.5:
            cv2.drawMarker(prediction_panel, tuple(np.rint(point).astype(int)), color,
                           cv2.MARKER_CROSS, 9, 2)
    row = np.hstack((
        label(mono, "current image"), label(truth_panel, "truth visible corners"),
        label(prediction_panel, "heatmap + visible predictions"),
    ))
    banner = np.zeros((48, row.shape[1], 3), np.uint8)
    title = f"{name.replace('_', ' ')} | {sample['trajectory_type']} | visible {candidate['visible_count']}/4"
    probabilities = "/".join(f"{value:.2f}" for value in visibility)
    detail = (
        f"error {candidate['mean_visible_error_px']:.1f}px | "
        f"distance {candidate['gate_distance_m']:.1f}m | q TL/TR/BR/BL {probabilities}"
    )
    cv2.putText(banner, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, detail, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (220, 220, 220), 1, cv2.LINE_AA)
    panel = np.vstack((banner, row))
    filename = f"{name}_{sample['trajectory_type']}.png"
    cv2.imwrite(str(output_dir / filename), panel)
    return {
        **candidate,
        "category": name,
        "file": filename,
        "layout_id": sample["layout_id"],
        "trajectory_id": sample["trajectory_id"],
        "trajectory_type": sample["trajectory_type"],
        "frame_index": int(sample["frame_index"]),
        "visibility_probability_tl_tr_br_bl": visibility.tolist(),
    }, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--maximum-gate-distance-m", type=float, default=8.0)
    parser.add_argument("--minimum-gate-span-px", type=float, default=16.0)
    parser.add_argument("--minimum-gate-area-px2", type=float, default=256.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this allocation")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    dataset = TTCGateDataset(
        args.dataset, "test",
        maximum_gate_distance_m=args.maximum_gate_distance_m,
        minimum_gate_span_px=args.minimum_gate_span_px,
        minimum_gate_area_px2=args.minimum_gate_area_px2,
    )
    model, initialization = load_dory_checkpoint(args.checkpoint, device)
    model.eval()
    candidates = scan(model, dataset, device, args.batch_size, args.workers)
    order = (
        "no_gate_correct", "one_corner", "two_corners", "three_corners", "full_gate_good",
        "full_gate_hard", "no_gate_false_alarm",
    )
    records, panels = [], []
    for name in order:
        if name in candidates:
            record, panel = render(model, dataset, name, candidates[name], device, args.output_dir)
            records.append(record)
            panels.append(panel)
    if panels:
        legend = np.zeros((30, panels[0].shape[1], 3), np.uint8)
        cv2.putText(legend, "Corners: TL blue | TR yellow | BR magenta | BL green",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output_dir / "gate_predictions_montage.png"),
                    np.vstack((legend, *panels)))
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization": initialization,
        "corner_order": ["TL", "TR", "BR", "BL"],
        "examples": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
