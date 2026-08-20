#!/usr/bin/env python3
"""Generate only the geometry labels required by the canonical 12-channel CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .gate_geometry import associate_gate
from .output_contract import NORMAL_ANGLES_DEG, OFFSET_MAX, OFFSET_MIN
from .rollout_targets import (
    course_collision_boxes,
    fixed_normal_offsets_to_boxes,
    load_calibration,
)
from .targets import class_mask, gate_opening_and_corners, reliable_gate_corner_view


SCHEMA = "sequential_fixed_normal_v1"


def _dimension(metadata: dict, key: str, fallback: float) -> float:
    value = metadata.get(key, fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("gap8_perception/configs/hm01b0_calibration.json"),
    )
    parser.add_argument("--drone-radius-m", type=float, default=0.10)
    parser.add_argument("--label-margin-m", type=float, default=0.10)
    args = parser.parse_args()

    if not (args.shard / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete shard: {args.shard}")
    metadata = json.loads((args.shard / "scene_metadata.json").read_text())
    poses = [
        json.loads(line)
        for line in (args.shard / "poses.jsonl").read_text().splitlines()
        if line
    ]
    monos = sorted(args.shard.glob("hm01b0_mono_*.png"))
    camera_matrix, distortion = load_calibration(args.calibration)
    boxes = course_collision_boxes(metadata)
    inner_m = _dimension(metadata, "gate_clear_opening_m", 0.45)
    outer_m = _dimension(metadata, "gate_outer_size_m", 0.66)
    centerline_m = (inner_m + outer_m) / 2.0
    gate_centers = np.asarray(
        metadata.get(
            "gate_centers_m",
            metadata.get("layout", {}).get(
                "gates", [(-1.30, -0.45, 0.55), (1.30, 0.45, 0.55)]
            ),
        ),
        np.float64,
    )

    count = len(monos)
    corners = np.full((count, 4, 2), np.nan, np.float16)
    corner_visibility = np.zeros((count, 4), np.uint8)
    offsets = np.full((count, 4), OFFSET_MAX, np.float16)
    offset_valid = np.zeros((count, 4), np.uint8)
    global_indices = np.zeros(count, np.int32)
    gate_indices = np.full(count, -1, np.int8)
    object_corners = np.full((count, 4, 3), np.nan, np.float32)
    projection_error = np.full(count, np.nan, np.float16)

    for row, mono in enumerate(monos):
        local = int(mono.stem.rsplit("_", 1)[1])
        pose = poses[local]
        suffix = f"{local:04d}"
        gate_mask = class_mask(
            args.shard / f"semantic_segmentation_{suffix}.png",
            args.shard / f"semantic_segmentation_labels_{suffix}.json",
            {"gate"},
        )
        _opening, inner_xy, raster_valid = gate_opening_and_corners(gate_mask)
        if raster_valid:
            gate_index, projection, error = associate_gate(
                inner_xy,
                pose["eye_m"],
                pose["target_m"],
                camera_matrix,
                distortion,
                inner_opening_m=inner_m,
                outer_edge_m=outer_m,
                gate_centers_world=gate_centers,
            )
            label_xy = projection["pixels_ordered"].astype(np.float32)
            visible = (
                (label_xy[:, 0] >= 0)
                & (label_xy[:, 0] < 160)
                & (label_xy[:, 1] >= 20)
                & (label_xy[:, 1] < 140)
            )
            if reliable_gate_corner_view(label_xy):
                corners[row] = label_xy.astype(np.float16)
                corner_visibility[row] = visible.astype(np.uint8) * 255
                gate_indices[row] = gate_index
                object_corners[row] = projection["object_points_ordered"]
                projection_error[row] = error

        current_offsets, current_valid = fixed_normal_offsets_to_boxes(
            pose["eye_m"],
            pose["target_m"],
            camera_matrix,
            distortion,
            boxes,
            args.drone_radius_m + args.label_margin_m,
            angles_deg=NORMAL_ANGLES_DEG,
            max_range_m=OFFSET_MAX,
            image_width=160,
            image_height=120,
        )
        offsets[row] = np.clip(current_offsets, OFFSET_MIN, OFFSET_MAX)
        offset_valid[row] = current_valid.astype(np.uint8) * 255
        global_indices[row] = pose["global_index"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(SCHEMA),
        corners_xy160_f16=corners,
        corner_visibility_u8=corner_visibility,
        corner_valid_u8=(corner_visibility.all(axis=1)).astype(np.uint8),
        fixed_normal_offsets_m_f16=offsets,
        fixed_normal_confidence_u8=offset_valid,
        fixed_normal_angles_deg_f32=np.asarray(NORMAL_ANGLES_DEG, np.float32),
        global_indices_i32=global_indices,
        gate_index_i8=gate_indices,
        gate_object_corners_m_f32=object_corners,
        gate_projection_error_px_f16=projection_error,
        corner_label_convention=np.asarray("gate_frame_centerline"),
        corner_label_extent_m_f32=np.asarray(centerline_m, np.float32),
        drone_radius_m_f32=np.asarray(args.drone_radius_m, np.float32),
        label_margin_m_f32=np.asarray(args.label_margin_m, np.float32),
    )
    summary = {
        "schema": SCHEMA,
        "shard": args.shard.name,
        "images": count,
        "corner_label_convention": "gate_frame_centerline",
        "corner_label_extent_m": centerline_m,
        "fully_visible_corner_frames": int(corner_visibility.all(axis=1).sum()),
        "fixed_normal_angles_deg": list(NORMAL_ANGLES_DEG),
        "offset_range_m": [float(offsets.min()), float(offsets.max())],
        "offset_mean_m": offsets.astype(np.float32).mean(axis=0).tolist(),
        "offset_observable_fraction": (offset_valid.mean(axis=0) / 255.0).tolist(),
        "inflation": {
            "drone_radius_m": args.drone_radius_m,
            "label_margin_m": args.label_margin_m,
        },
        "excluded_legacy_targets": [
            "speed-indexed rollout maps",
            "dense danger",
            "inverse range",
            "urgency",
            "uncertainty",
            "vehicle state",
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
