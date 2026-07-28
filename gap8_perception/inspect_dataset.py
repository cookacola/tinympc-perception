#!/usr/bin/env python3
"""Audit the existing 50k dataset and visualize 100 deterministic samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from gap8_perception.targets import (
    COLLISION_CLASSES,
    class_mask,
    danger_target,
    gate_opening_and_corners,
)


def sample_records(root: Path):
    for shard in sorted(root.glob("shard_*")):
        if not (shard / "_SUCCESS").is_file():
            continue
        poses = {
            row["local_index"]: row
            for row in map(json.loads, (shard / "poses.jsonl").read_text().splitlines())
        }
        for mono in sorted(shard.glob("hm01b0_mono_*.png")):
            local = int(mono.stem.rsplit("_", 1)[1])
            suffix = f"{local:04d}"
            yield {
                "shard": shard,
                "local": local,
                "global": poses[local]["global_index"],
                "pose": poses[local],
                "mono": mono,
                "rgb": shard / f"rgb_{suffix}.png",
                "depth": shard / f"depth_mm_{suffix}.png",
                "semantic": shard / f"semantic_segmentation_{suffix}.png",
                "labels": shard / f"semantic_segmentation_labels_{suffix}.json",
            }


def overlay(record):
    mono = cv2.imread(str(record["mono"]), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(record["depth"]), cv2.IMREAD_UNCHANGED)
    gate = class_mask(record["semantic"], record["labels"], {"gate"})
    collision = class_mask(record["semantic"], record["labels"], COLLISION_CLASSES)
    opening, corners, valid = gate_opening_and_corners(gate)
    danger = danger_target(depth, collision)
    danger160 = cv2.resize(danger, (160, 160), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    image[danger160 > 0.5] = (
        0.45 * image[danger160 > 0.5] + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)
    image[opening > 0] = (
        0.35 * image[opening > 0] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)
    if valid:
        colors = ((255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255))
        for point, color in zip(corners, colors):
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1)
    cv2.putText(
        image,
        f"g={record['global']} corners={int(valid)}",
        (3, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image, gate, opening, danger, valid, depth


def rollout_overlay(record, archive):
    """Show all control targets for one image in a compact 320x160 tile."""
    mono = cv2.imread(str(record["mono"]), cv2.IMREAD_GRAYSCALE)
    gate = class_mask(record["semantic"], record["labels"], {"gate"})
    opening, corners, valid = gate_opening_and_corners(gate)
    image = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    image[opening > 0] = (
        0.35 * image[opening > 0] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)
    if valid:
        colors = ((255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255))
        for point, color in zip(corners, colors):
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1)
    local = record["local"]
    danger = archive["danger_u8"][local]
    inverse_range = archive["inverse_range_u8"][local]
    clearance = archive["minimum_clearance_m_f16"][local, 1].astype(np.float32)
    ttc = archive["time_to_collision_s_f16"][local, 1].astype(np.float32)
    ttc_normalized = np.where(ttc >= 0, np.clip(ttc / 1.08, 0, 1), 1.0)
    clearance_normalized = np.clip((clearance + 0.2) / 1.2, 0, 1)
    maps = [
        (danger[0], "D .5"),
        (danger[1], "D 1"),
        (danger[2], "D 5"),
        (inverse_range, "INV R"),
        (np.rint(clearance_normalized * 255).astype(np.uint8), "CLR"),
        (np.rint(ttc_normalized * 255).astype(np.uint8), "TTC"),
    ]
    small = []
    for values, label in maps:
        colored = cv2.applyColorMap(
            cv2.resize(values, (53, 80), interpolation=cv2.INTER_NEAREST),
            cv2.COLORMAP_TURBO,
        )
        cv2.putText(
            colored, label, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        small.append(colored)
    right = np.vstack((np.hstack(small[:3]), np.hstack(small[3:])))
    return np.hstack((image, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--targets", type=Path,
        help="Cached rollout-target root; adds a contact sheet for all control labels.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = list(sample_records(args.dataset))
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.samples, len(records)))

    stats = Counter()
    hashes = Counter()
    tiles = []
    for record in selected:
        image, gate, opening, danger, valid, depth = overlay(record)
        tiles.append(image)
        stats["corner_valid"] += int(valid)
        stats["gate_visible"] += int(gate.any())
        stats["gate_pixels"] += int(gate.sum())
        stats["opening_pixels"] += int(opening.sum())
        stats["danger_pixels"] += int((danger >= 0.5).sum())
        stats["danger_total"] += danger.size
        stats["invalid_depth_pixels"] += int((depth == 0).sum())
        stats["depth_pixels"] += depth.size
        hashes[hashlib.sha256(record["mono"].read_bytes()).hexdigest()] += 1

    rows = []
    for start in range(0, len(tiles), 10):
        row = tiles[start : start + 10]
        row += [np.zeros_like(tiles[0])] * (10 - len(row))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.imwrite(str(args.output / "inspection_100.jpg"), sheet)

    if args.targets is not None:
        archives = {}
        rollout_tiles = []
        for record in selected:
            shard_name = record["shard"].name
            if shard_name not in archives:
                loaded = np.load(args.targets / f"{shard_name}.npz")
                archives[shard_name] = {
                    key: loaded[key] for key in loaded.files
                }
                loaded.close()
            rollout_tiles.append(
                rollout_overlay(record, archives[shard_name])
            )
        rollout_rows = [
            np.hstack(rollout_tiles[start : start + 5])
            for start in range(0, len(rollout_tiles), 5)
        ]
        cv2.imwrite(
            str(args.output / "inspection_rollout_100.jpg"),
            np.vstack(rollout_rows),
        )

    report = {
        "dataset": str(args.dataset.resolve()),
        "frames": len(records),
        "successful_shards": len({str(r["shard"]) for r in records}),
        "sampled": len(selected),
        "semantic_corner_order": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ],
        "corner_source": "largest enclosed hole in semantic gate-frame raster",
        "corner_valid_sampled": stats["corner_valid"],
        "gate_visible_sampled": stats["gate_visible"],
        "gate_pixel_fraction_sampled": stats["gate_pixels"] / (len(selected) * 160 * 160),
        "opening_pixel_fraction_sampled": stats["opening_pixels"]
        / (len(selected) * 160 * 160),
        "danger_positive_fraction_sampled": stats["danger_pixels"]
        / stats["danger_total"],
        "invalid_depth_fraction_sampled": stats["invalid_depth_pixels"]
        / stats["depth_pixels"],
        "duplicate_mono_hashes_sampled": sum(v - 1 for v in hashes.values() if v > 1),
        "available": [
            "rgb uint8 PNG 160x160",
            "HM01B0-like mono uint8 PNG 160x160",
            "depth uint16 PNG in millimeters",
            "class semantic uint8 PNG + per-frame raw-ID JSON",
            "camera eye/look-at pose and global index",
            "fixed scene metadata",
            "derived shard seed = generator base seed + shard start index",
        ],
        "missing": [
            "stored camera intrinsics/distortion",
            "gate instance IDs and authoritative 2D/3D corners",
            "per-frame gate poses",
            "vehicle state/velocity/attitude",
            "timestamps and trajectories",
            "multiple scene identities",
            "rollout collision outcomes",
        ],
        "split_policy": "shard-level simulator-seed split; one fixed scene, no trajectory claims",
        "danger_target_status": (
            "exact calibrated-ray swept-sphere rollout against fixed-scene "
            "AABBs at synthetic 0.5/1.0/5.0 m/s state variants"
            if args.targets is not None
            else "source-only depth-band visualization; pass --targets for final rollout labels"
        ),
        "rollout_annotation_sheet": (
            "inspection_rollout_100.jpg" if args.targets is not None else None
        ),
    }
    (args.output / "dataset_inspection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
