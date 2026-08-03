#!/usr/bin/env python3
"""Render design-correct corner and fixed-normal ground-truth overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .output_contract import NORMAL_ANGLES_DEG, OFFSET_MAX


COLORS = ((255, 80, 40), (0, 240, 255), (255, 0, 220), (0, 150, 255))


def title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 15), (0, 0, 0), -1)
    cv2.putText(output, text, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def direction_panel(offsets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    panel = np.full((120, 160, 3), 24, np.uint8)
    origin = np.asarray((80.0, 112.0))
    cv2.circle(panel, tuple(origin.astype(int)), 4, (255, 255, 255), -1)
    for index, (angle, distance) in enumerate(zip(NORMAL_ANGLES_DEG, offsets)):
        radians = np.radians(angle)
        unit = np.asarray((-np.sin(radians), -np.cos(radians)))
        endpoint = origin + unit * float(distance) * 15.0
        color = (40, 220, 40) if valid[index] else (100, 100, 100)
        cv2.arrowedLine(panel, tuple(origin.astype(int)), tuple(np.rint(endpoint).astype(int)),
                        color, 2, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(panel, f'{angle:g}:{distance:.2f}m',
                    (3, 29 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    COLORS[index], 1, cv2.LINE_AA)
    return title(panel, 'GT fixed body normals (top view)')


def bar_panel(offsets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    panel = np.full((120, 160, 3), 24, np.uint8)
    for index, (distance, observable) in enumerate(zip(offsets, valid)):
        y = 25 + index * 22
        width = int(np.clip(distance / OFFSET_MAX, 0, 1) * 112)
        cv2.rectangle(panel, (40, y), (40 + width, y + 11),
                      COLORS[index] if observable else (90, 90, 90), -1)
        cv2.putText(panel, f'n{index}', (4, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f'{distance:.2f}', (122, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (255, 255, 255), 1, cv2.LINE_AA)
    return title(panel, 'GT free distance (0-6 m)')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.targets) as target:
        valid = target['corner_valid_u8'].astype(bool)
        preferred = np.flatnonzero(valid).tolist()
        fallback = np.flatnonzero(~valid).tolist()
        selected = (preferred + fallback)[:args.samples]
        rows = []
        for index in selected:
            image = cv2.imread(str(args.shard / f'hm01b0_mono_{index:04d}.png'))
            crop = image[20:140]
            corner = crop.copy()
            corners = target['corners_xy160_f16'][index].astype(np.float32)
            visibility = target['corner_visibility_u8'][index].astype(bool)
            crop_corners = corners - np.asarray((0, 20), np.float32)
            if visibility.all():
                overlay = corner.copy()
                cv2.fillPoly(overlay, [np.rint(crop_corners).astype(np.int32)], (0, 180, 0))
                corner = cv2.addWeighted(corner, 0.55, overlay, 0.45, 0)
            for channel in np.flatnonzero(visibility):
                point = tuple(np.rint(crop_corners[channel]).astype(int))
                cv2.circle(corner, point, 4, COLORS[channel], -1)
                cv2.putText(corner, ('TL', 'TR', 'BR', 'BL')[channel],
                            (point[0] + 3, point[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, COLORS[channel], 1, cv2.LINE_AA)
            offsets = target['fixed_normal_offsets_m_f16'][index].astype(np.float32)
            observable = target['fixed_normal_confidence_u8'][index].astype(bool)
            row = cv2.hconcat([
                title(crop, f'frame {index} input'),
                title(corner, 'GT centerline corners'),
                direction_panel(offsets, observable),
                bar_panel(offsets, observable),
            ])
            rows.append(row)
            cv2.imwrite(str(args.output / f'ground_truth_{index:04d}.png'), row)
    montage = cv2.vconcat(rows)
    cv2.imwrite(str(args.output / 'sequential_design_ground_truth_montage.png'), montage)
    print(args.output / 'sequential_design_ground_truth_montage.png')


if __name__ == '__main__':
    main()
