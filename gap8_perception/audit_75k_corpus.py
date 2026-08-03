#!/usr/bin/env python3
"""Verify the 75k scenario corpus before any training job consumes it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def shard_names(spec: dict | list[str]) -> list[str]:
    if isinstance(spec, list):
        return spec
    return [
        f"shard_{index:09d}"
        for index in range(int(spec["start"]), int(spec["stop"]), int(spec["step"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-gate-frame-pixels", type=int, default=180,
        help="reject a fresh training frame whose visible gate is too small",
    )
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text())
    expected = {name: shard_names(split[name]) for name in ("train", "validation", "test")}
    seen = set().union(*map(set, expected.values()))
    if sum(map(len, expected.values())) != len(seen):
        raise RuntimeError("split file reuses one or more episode shards")

    report: dict[str, object] = {
        "splits": {}, "scenario_frames": Counter(),
        "minimum_gate_frame_pixels": args.min_gate_frame_pixels,
    }
    total_frames = 0
    for split_name, names in expected.items():
        split_frames = 0
        split_scenarios: Counter[str] = Counter()
        for name in names:
            shard = args.dataset / name
            target = args.targets / f"{name}.npz"
            if not (shard / "_SUCCESS").is_file() or not target.is_file():
                raise RuntimeError(f"incomplete shard/target pair: {name}")
            completion = json.loads((shard / "_SUCCESS").read_text())
            if completion.get("fresh_render") is not True:
                raise RuntimeError(f"{name} is not attested as a fresh render")
            if shard.resolve().parent != args.dataset.resolve():
                raise RuntimeError(f"{name} is not a direct shard of the fresh dataset root")
            metadata = json.loads((shard / "scene_metadata.json").read_text())
            sensor = json.loads((shard / "camera_sensor.json").read_text())
            scenario = metadata.get("scenario")
            if scenario not in {"calibrated", "randomized", "hard_negative"}:
                raise RuntimeError(f"{name} lacks a valid scenario")
            if not metadata.get("collision_obstacles"):
                raise RuntimeError(f"{name} lacks recorded collision obstacles")
            if sensor.get("sensor_profile") != "flight08_calibrated_v1":
                raise RuntimeError(f"{name} lacks the calibrated HM01B0 profile")
            if metadata.get("camera_rig_occlusion", {}).get("geometry") != (
                "single calibrated sensor-stage silhouette matched to stream_out"
            ):
                raise RuntimeError(f"{name} lacks the single calibrated guard model")
            constraints = metadata.get("camera_pose_constraints", {})
            if not constraints.get("require_visible_gate"):
                raise RuntimeError(f"{name} does not require visible near-gate views")
            if int(constraints.get("min_gate_frame_pixels", -1)) < args.min_gate_frame_pixels:
                raise RuntimeError(f"{name} has a weaker gate-area constraint")
            frames = len(list(shard.glob("hm01b0_mono_*.png")))
            if frames == 0:
                raise RuntimeError(f"{name} has no HM01B0 frames")
            gate_pixels = []
            for semantic_path in sorted(
                shard.glob("semantic_segmentation_[0-9][0-9][0-9][0-9].png")
            ):
                labels_path = semantic_path.with_name(
                    semantic_path.name.replace(
                        "semantic_segmentation_", "semantic_segmentation_labels_"
                    )
                ).with_suffix(".json")
                labels = json.loads(labels_path.read_text())
                gate_ids = [
                    int(raw) for raw, fields in labels.items()
                    if str(fields.get("class", "")).lower() == "gate"
                ]
                semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
                gate_pixels.append(int(np.isin(semantic, gate_ids).sum()))
            if len(gate_pixels) != frames or min(gate_pixels) < args.min_gate_frame_pixels:
                raise RuntimeError(
                    f"{name} contains a distant/absent gate: min pixels="
                    f"{min(gate_pixels, default=0)}"
                )
            with np.load(target) as archive:
                if str(archive["schema"].item()) != "sequential_fixed_normal_v1":
                    raise RuntimeError(f"{name} is not a canonical target archive")
                if archive["fixed_normal_offsets_m_f16"].shape != (frames, 4):
                    raise RuntimeError(f"{name} offset-target shape mismatch")
                if archive["fixed_normal_confidence_u8"].shape != (frames, 4):
                    raise RuntimeError(f"{name} confidence-target shape mismatch")
                if archive["corner_visibility_u8"].shape != (frames, 4):
                    raise RuntimeError(f"{name} corner-visibility shape mismatch")
                if any(key in archive for key in ("danger_u8", "speed_variants_mps", "vehicle_state_f32")):
                    raise RuntimeError(f"{name} contains legacy speed/danger targets")
            split_frames += frames
            split_scenarios[scenario] += frames
            report["scenario_frames"][scenario] += frames
        report["splits"][split_name] = {
            "episodes": len(names), "frames": split_frames,
            "scenario_frames": dict(split_scenarios),
        }
        total_frames += split_frames

    report["total_frames"] = total_frames
    report["scenario_frames"] = dict(report["scenario_frames"])
    expected_frames = {"calibrated": 45000, "randomized": 22500, "hard_negative": 7500}
    if total_frames != 75000 or report["scenario_frames"] != expected_frames:
        raise RuntimeError(f"unexpected corpus allocation: {report}")
    if [report["splits"][name]["frames"] for name in ("train", "validation", "test")] != [60000, 7500, 7500]:
        raise RuntimeError(f"unexpected split allocation: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
