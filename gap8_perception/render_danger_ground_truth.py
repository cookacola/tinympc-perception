#!/usr/bin/env python3
"""Render an unambiguous collision-target example for human inspection."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np

from gap8_perception.targets import class_mask


def labeled(panel: np.ndarray, text: str) -> np.ndarray:
    output = panel.copy()
    lines = textwrap.wrap(text, width=43)[:2]
    bar_height = 16 + 14 * len(lines)
    cv2.rectangle(output, (0, 0), (output.shape[1], bar_height), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(
            output, line, (6, 15 + 14 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-obstacle",
        action="store_true",
        help="Select a frame with a large simulator-labeled obstacle in front",
    )
    args = parser.parse_args()

    choices = []
    for target_path in sorted(args.targets.glob("shard_*.npz")):
        with np.load(target_path) as target:
            danger = target["danger_u8"][:, 1] >= 128
            fractions = danger.mean(axis=(1, 2))
            # Prefer a structured map near 35% positive, not an all-safe/all-red
            # example. Boundary count rewards visible spatial structure.
            boundaries = (
                (danger[:, 1:] != danger[:, :-1]).sum(axis=(1, 2))
                + (danger[:, :, 1:] != danger[:, :, :-1]).sum(axis=(1, 2))
            )
            scores = -np.abs(fractions - 0.35) + boundaries / 1000.0
            candidate_count = min(20, len(scores))
            for local in np.argpartition(scores, -candidate_count)[-candidate_count:]:
                score = float(scores[local])
                if args.require_obstacle:
                    shard = args.dataset / target_path.stem
                    semantic = shard / f"semantic_segmentation_{int(local):04d}.png"
                    labels = (
                        shard
                        / f"semantic_segmentation_labels_{int(local):04d}.json"
                    )
                    obstacle = class_mask(semantic, labels, {"obstacle"})
                    yy, xx = np.mgrid[:160, :160]
                    center_weight = np.exp(
                        -((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 45**2)
                    )
                    obstacle_score = float((obstacle * center_weight).sum())
                    if obstacle_score < 100:
                        continue
                    score += obstacle_score / 1000.0
                choices.append((score, target_path, int(local)))
    if not choices:
        raise RuntimeError("no frame satisfies the requested ground-truth view")
    _, target_path, local = max(choices)
    shard = args.dataset / target_path.stem
    mono_path = shard / f"hm01b0_mono_{local:04d}.png"
    with np.load(target_path) as target:
        collision = target["danger_u8"][local, 1].astype(np.float32) / 255.0
        inverse_range = (
            target["inverse_range_u8"][local].astype(np.float32) / 255.0
        )
        clearance = target["minimum_clearance_m_f16"][local, 1].astype(np.float32)

    mono = cv2.imread(str(mono_path), cv2.IMREAD_GRAYSCALE)
    if mono is None:
        raise FileNotFoundError(mono_path)
    base = cv2.cvtColor(cv2.resize(mono, (320, 320)), cv2.COLOR_GRAY2BGR)
    collision320 = cv2.resize(collision, (320, 320), cv2.INTER_NEAREST)
    binary_u8 = np.uint8(collision320 * 255)
    binary = cv2.applyColorMap(binary_u8, cv2.COLORMAP_JET)
    overlay = base.copy()
    positive = collision320 >= 0.5
    overlay[positive] = (
        0.35 * overlay[positive] + 0.65 * np.array([0, 0, 255])
    ).astype(np.uint8)
    for coordinate in range(0, 321, 16):
        cv2.line(overlay, (coordinate, 0), (coordinate, 320), (80, 80, 80), 1)
        cv2.line(overlay, (0, coordinate), (320, coordinate), (80, 80, 80), 1)
    inv = cv2.applyColorMap(
        np.uint8(cv2.resize(inverse_range, (320, 320), cv2.INTER_NEAREST) * 255),
        cv2.COLORMAP_VIRIDIS,
    )
    clearance_vis = np.clip((clearance + 0.2) / 1.5, 0, 1)
    clearance_panel = cv2.applyColorMap(
        np.uint8(
            cv2.resize(clearance_vis, (320, 320), cv2.INTER_NEAREST) * 255
        ),
        cv2.COLORMAP_VIRIDIS,
    )
    fraction = float((collision >= 0.5).mean())
    sheet = np.vstack(
        (
            np.hstack(
                (
                    labeled(base, f"HM01B0 input: {target_path.stem}/{local:04d}"),
                    labeled(
                        binary,
                        f"GROUND TRUTH collision @ 1.0 m/s ({fraction:.1%} positive)",
                    ),
                )
            ),
            np.hstack(
                (
                    labeled(overlay, "Ground-truth overlay: RED = collision within 1.0 s"),
                    labeled(inv, "Auxiliary ground truth: inverse range (bright = near)"),
                )
            ),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"failed to write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
