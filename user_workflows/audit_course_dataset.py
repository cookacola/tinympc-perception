#!/usr/bin/env python3
"""Audit a complete sharded course dataset against global invariants."""

import argparse
import json
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--expected-frames", type=int, default=50000)
parser.add_argument("--allow-incomplete", action="store_true")
args = parser.parse_args()

errors = []
global_indices = set()
modality_counts = Counter()
verified_frames = 0
successful_shards = []
for shard in sorted(args.root.glob("shard_*")):
    if not (shard / "_SUCCESS").is_file():
        continue
    successful_shards.append(shard.name)
    validation = json.loads((shard / "validation.json").read_text())
    if not validation.get("passed"):
        errors.append(f"{shard.name}: validation did not pass")
        continue
    expected = int(validation["expected"])
    verified_frames += expected

    patterns = {
        "rgb": "rgb_*.png",
        "hm01b0_mono": "hm01b0_mono_*.png",
        "depth_mm": "depth_mm_*.png",
        "semantic": "semantic_segmentation_[0-9][0-9][0-9][0-9].png",
        "semantic_labels": "semantic_segmentation_labels_*.json",
    }
    for modality, pattern in patterns.items():
        count = len(list(shard.glob(pattern)))
        modality_counts[modality] += count
        if count != expected:
            errors.append(f"{shard.name}: {modality} count={count}, expected={expected}")

    poses = [json.loads(line) for line in (shard / "poses.jsonl").read_text().splitlines()]
    if len(poses) != expected:
        errors.append(f"{shard.name}: pose count={len(poses)}, expected={expected}")
    for pose in poses:
        index = int(pose["global_index"])
        if index in global_indices:
            errors.append(f"duplicate global index {index}")
        global_indices.add(index)
        if pose.get("resolution") != [160, 160]:
            errors.append(f"{shard.name}: index {index} is not 160x160")

    scene = json.loads((shard / "scene_metadata.json").read_text())
    required_scene = {
        "course_count": 1,
        "course_extent_m": [4.0, 4.0],
        "obstacle_count": 2,
        "gate_count": 2,
        "gate_model": "NewBeeDrone Micro Race Gate - Square",
    }
    for key, value in required_scene.items():
        if scene.get(key) != value:
            errors.append(f"{shard.name}: scene {key}={scene.get(key)!r}, expected={value!r}")
    if scene.get("gate_texture", {}).get("version") != "newbeedrone_hm01b0_v2":
        errors.append(f"{shard.name}: wrong or missing NewBeeDrone texture version")

    sensor = json.loads((shard / "camera_sensor.json").read_text())
    if sensor.get("model") != "Himax HM01B0" or sensor.get("resolution") != [160, 160]:
        errors.append(f"{shard.name}: wrong camera sensor metadata")

if verified_frames != args.expected_frames and not args.allow_incomplete:
    errors.append(
        f"verified frame count={verified_frames}, expected={args.expected_frames}"
    )
if verified_frames and len(global_indices) != verified_frames:
    errors.append(
        f"unique global indices={len(global_indices)}, verified frames={verified_frames}"
    )
if not args.allow_incomplete and global_indices != set(range(args.expected_frames)):
    missing = sorted(set(range(args.expected_frames)) - global_indices)
    errors.append(f"global index coverage incomplete; first missing={missing[:10]}")

report = {
    "passed": not errors,
    "complete": verified_frames == args.expected_frames,
    "expected_frames": args.expected_frames,
    "verified_frames": verified_frames,
    "successful_shards": len(successful_shards),
    "unique_global_indices": len(global_indices),
    "modality_counts": dict(modality_counts),
    "errors": errors,
}
(args.root / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if not errors else 1)
