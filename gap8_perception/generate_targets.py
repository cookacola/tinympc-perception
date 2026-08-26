#!/usr/bin/env python3
"""Generate cached rollout danger, urgency, gate, and corner targets by shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gap8_perception.rollout_targets import (
    load_calibration,
    simulate_candidate_rollouts,
)
from gap8_perception.gate_geometry import associate_gate
from gap8_perception.targets import class_mask, gate_opening_and_corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("gap8_perception/configs/hm01b0_calibration.json"),
    )
    parser.add_argument("--drone-radius-m", type=float, default=0.10)
    parser.add_argument("--safety-margin-m", type=float, default=0.10)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--timestep-s", type=float, default=0.05)
    parser.add_argument("--latency-s", type=float, default=0.08)
    parser.add_argument("--acceleration-limit-mps2", type=float, default=6.0)
    parser.add_argument("--attitude-limit-deg", type=float, default=35.0)
    args = parser.parse_args()
    if not (args.shard / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete shard: {args.shard}")

    monos = sorted(args.shard.glob("hm01b0_mono_*.png"))
    speeds = (0.5, 1.0, 5.0)
    variants = len(speeds)
    danger = np.zeros((len(monos), variants, 20, 20), dtype=np.uint8)
    urgency = np.zeros_like(danger)
    uncertainty = np.zeros_like(danger)
    inverse_range = np.zeros((len(monos), 20, 20), dtype=np.uint8)
    obstacle_presence = np.zeros((len(monos), 20, 20), dtype=np.uint8)
    conservative_range = np.zeros((len(monos), 20, 20), dtype=np.float16)
    minimum_clearance = np.zeros(
        (len(monos), variants, 20, 20), dtype=np.float16
    )
    time_to_collision = np.full_like(minimum_clearance, -1)
    vehicle_state = np.zeros((len(monos), variants, 8), dtype=np.float32)
    gate_opening = np.zeros((len(monos), 40, 40), dtype=np.uint8)
    gate_opening_sdf = np.zeros((len(monos), 40, 40), dtype=np.float16)
    corners = np.full((len(monos), 4, 2), np.nan, dtype=np.float16)
    corner_valid = np.zeros(len(monos), dtype=np.uint8)
    global_indices = np.zeros(len(monos), dtype=np.int32)
    gate_index = np.full(len(monos), -1, dtype=np.int8)
    gate_projected_corners = np.full((len(monos), 4, 2), np.nan, dtype=np.float16)
    gate_object_corners = np.full((len(monos), 4, 3), np.nan, dtype=np.float32)
    gate_projection_error = np.full(len(monos), np.nan, dtype=np.float16)
    gate_translation_camera = np.full((len(monos), 3), np.nan, dtype=np.float32)
    gate_rotation_camera = np.full((len(monos), 3, 3), np.nan, dtype=np.float32)
    poses = [
        json.loads(line)
        for line in (args.shard / "poses.jsonl").read_text().splitlines()
    ]
    camera_matrix, distortion = load_calibration(args.calibration)

    for row, mono in enumerate(monos):
        local = int(mono.stem.rsplit("_", 1)[1])
        suffix = f"{local:04d}"
        semantic = args.shard / f"semantic_segmentation_{suffix}.png"
        labels = args.shard / f"semantic_segmentation_labels_{suffix}.json"
        gate_frame = class_mask(semantic, labels, {"gate"})
        opening, xy, valid = gate_opening_and_corners(gate_frame)
        gate_opening[row] = cv2.resize(
            opening, (40, 40), interpolation=cv2.INTER_AREA
        ) > 0
        inside = gate_opening[row].astype(np.uint8)
        signed_distance = (
            cv2.distanceTransform(inside, cv2.DIST_L2, 5)
            - cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
        )
        gate_opening_sdf[row] = np.clip(signed_distance / 8.0, -1.0, 1.0)
        corners[row] = xy.astype(np.float16)
        corner_valid[row] = valid
        global_indices[row] = poses[local]["global_index"]
        if valid:
            associated, projection, projection_error = associate_gate(
                xy,
                poses[local]["eye_m"],
                poses[local]["target_m"],
                camera_matrix,
                distortion,
            )
            gate_index[row] = associated
            gate_projected_corners[row] = projection["pixels_ordered"].astype(
                np.float16
            )
            gate_object_corners[row] = projection[
                "object_points_ordered"
            ].astype(np.float32)
            gate_projection_error[row] = projection_error
            gate_translation_camera[row] = projection[
                "translation_camera_from_gate"
            ].astype(np.float32)
            gate_rotation_camera[row] = projection[
                "rotation_camera_from_gate"
            ].astype(np.float32)

        for variant, speed in enumerate(speeds):
            rollout = simulate_candidate_rollouts(
                poses[local]["eye_m"],
                poses[local]["target_m"],
                camera_matrix,
                distortion,
                speed,
                horizon_s=args.horizon_s,
                timestep_s=args.timestep_s,
                latency_s=args.latency_s,
                drone_radius_m=args.drone_radius_m,
                safety_margin_m=args.safety_margin_m,
                acceleration_limit_mps2=args.acceleration_limit_mps2,
                attitude_limit_deg=args.attitude_limit_deg,
            )
            danger[row, variant] = rollout["collision"] * 255
            urgency[row, variant] = np.rint(
                rollout["urgency"] * 255
            ).astype(np.uint8)
            uncertainty[row, variant] = np.rint(
                rollout["uncertainty"] * 255
            ).astype(np.uint8)
            minimum_clearance[row, variant] = rollout[
                "minimum_clearance_m"
            ].astype(np.float16)
            time_to_collision[row, variant] = rollout["ttc_s"].astype(
                np.float16
            )
            # State convention: body +X forward, +Y left, +Z up; angular
            # velocity is zero because the source capture omitted vehicle state.
            vehicle_state[row, variant] = [
                speed,
                0,
                0,
                0,
                0,
                0,
                args.horizon_s,
                args.latency_s,
            ]
            if variant == 0:
                inverse_range[row] = np.rint(
                    rollout["inverse_range"] * 255
                ).astype(np.uint8)
                conservative_range[row] = rollout["range_m"].astype(np.float16)
                obstacle_presence[row] = (
                    rollout["range_m"] < 6.0 - 1e-3
                ).astype(np.uint8) * 255

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        danger_u8=danger,
        collision_within_horizon_u8=danger,
        urgency_u8=urgency,
        inverse_range_u8=inverse_range,
        obstacle_presence_u8=obstacle_presence,
        conservative_range_m_f16=conservative_range,
        uncertainty_u8=uncertainty,
        minimum_clearance_m_f16=minimum_clearance,
        time_to_collision_s_f16=time_to_collision,
        vehicle_state_f32=vehicle_state,
        speed_variants_mps=np.asarray(speeds, np.float32),
        gate_opening_u8=gate_opening,
        gate_opening_sdf_f16=gate_opening_sdf,
        corners_xy160_f16=corners,
        corner_valid_u8=corner_valid,
        global_indices_i32=global_indices,
        gate_index_i8=gate_index,
        gate_projected_corners_xy160_f16=gate_projected_corners,
        gate_object_corners_m_f32=gate_object_corners,
        gate_projection_error_px_f16=gate_projection_error,
        gate_translation_camera_m_f32=gate_translation_camera,
        gate_rotation_camera_f32=gate_rotation_camera,
    )
    summary = {
        "shard": args.shard.name,
        "images": len(monos),
        "state_variants_per_image": variants,
        "training_records": len(monos),
        "corner_valid_images": int(corner_valid.sum()),
        "danger_positive_fraction": float((danger >= 128).mean()),
        "danger_positive_fraction_by_speed": {
            str(speed): float((danger[:, variant] >= 128).mean())
            for variant, speed in enumerate(speeds)
        },
        "mean_urgency_by_speed": {
            str(speed): float(urgency[:, variant].mean() / 255.0)
            for variant, speed in enumerate(speeds)
        },
        "mean_inverse_range": float(inverse_range.mean() / 255.0),
        "range_target": (
            "first calibrated viewing-ray intersection with collision-volume-"
            "inflated scene AABBs, clipped at 6 m; independent of speed"
        ),
        "gate_opening_fraction": float(gate_opening.mean()),
        "danger_method": (
            "exact swept-sphere collision checking against fixed-scene AABBs "
            "for each calibrated viewing ray and candidate motion"
        ),
        "danger_assumptions": {
            "drone_radius_m": args.drone_radius_m,
            "safety_margin_m": args.safety_margin_m,
            "horizon_s": args.horizon_s,
            "timestep_s": args.timestep_s,
            "latency_s": args.latency_s,
            "acceleration_limit_mps2": args.acceleration_limit_mps2,
            "attitude_limit_deg": args.attitude_limit_deg,
            "state_variants_mps": speeds,
            "source_vehicle_state": (
                "synthetic state variants because original capture omitted "
                "vehicle motion metadata"
            ),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
