#!/usr/bin/env python3
"""Inspect every aligned RGB/depth/semantic annotation and render overlays."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--expected", type=int, default=100)
args = parser.parse_args()

root = args.dataset
overlay_dir = root / "inspection"
overlay_dir.mkdir(exist_ok=True)
palette = np.asarray(
    [[0, 0, 0], [96, 96, 96], [0, 165, 255], [255, 160, 0],
     [40, 40, 230], [40, 190, 40]],
    dtype=np.uint8,
)
errors, records, thumbnails = [], [], []
class_frames, class_pixels = Counter(), Counter()

for index in range(args.expected):
    suffix = f"{index:04d}"
    paths = {
        "rgb": root / f"rgb_{suffix}.png",
        "mono": root / f"hm01b0_mono_{suffix}.png",
        "depth": root / f"depth_mm_{suffix}.png",
        "semantic": root / f"semantic_segmentation_{suffix}.png",
        "labels": root / f"semantic_segmentation_labels_{suffix}.json",
    }
    missing = [kind for kind, path in paths.items() if not path.is_file()]
    if missing:
        errors.append(f"frame {index}: missing {','.join(missing)}")
        continue

    rgb = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    mono = cv2.imread(str(paths["mono"]), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    semantic = cv2.imread(str(paths["semantic"]), cv2.IMREAD_UNCHANGED)
    labels = json.loads(paths["labels"].read_text())
    if rgb is None or mono is None or depth is None or semantic is None:
        errors.append(f"frame {index}: unreadable modality")
        continue
    if rgb.shape[:2] != mono.shape or mono.shape != depth.shape or depth.shape != semantic.shape:
        errors.append(f"frame {index}: modality shape mismatch")
        continue
    if mono.dtype != np.uint8 or depth.dtype != np.uint16 or semantic.dtype != np.uint16:
        errors.append(f"frame {index}: expected uint8 mono and uint16 depth/semantics")
    expected_mono = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if not np.array_equal(mono, expected_mono):
        errors.append(f"frame {index}: HM01B0 stream is not pixel-aligned grayscale RGB")

    valid_ids = {int(key) for key in labels}
    ids, counts = np.unique(semantic, return_counts=True)
    unknown = set(int(value) for value in ids) - valid_ids
    if unknown:
        errors.append(f"frame {index}: semantic IDs absent from labels: {sorted(unknown)}")
    names = {
        int(key): str(value.get("class", "unknown")).lower()
        for key, value in labels.items()
    }
    per_frame = {}
    for class_id, count in zip(ids, counts):
        name = names.get(int(class_id), "unknown")
        per_frame[name] = int(count)
        class_frames[name] += 1
        class_pixels[name] += int(count)

    mean, variance = float(rgb.mean()), float(rgb.var())
    if mean <= 20 or variance <= 15:
        errors.append(f"frame {index}: unusable RGB mean={mean:.2f} variance={variance:.2f}")
    if not np.any(depth):
        errors.append(f"frame {index}: no valid depth")

    color_mask = palette[np.minimum(semantic, len(palette) - 1)]
    overlay = cv2.addWeighted(rgb, 0.65, color_mask, 0.35, 0)
    cv2.putText(
        overlay, f"{index:03d}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    output = overlay_dir / f"overlay_{suffix}.jpg"
    cv2.imwrite(str(output), overlay, [cv2.IMWRITE_JPEG_QUALITY, 86])
    thumbnails.append(cv2.resize(overlay, (160, 120)))
    records.append(
        {
            "frame": index,
            "rgb_mean": mean,
            "rgb_variance": variance,
            "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
            "semantic_pixels": per_frame,
            "overlay": str(output.relative_to(root)),
        }
    )

if class_frames["gate"] < max(1, args.expected // 5):
    errors.append(
        f"gate visible in only {class_frames['gate']}/{args.expected} frames; expected >=20%"
    )
if class_frames["obstacle"] < max(1, args.expected // 5):
    errors.append(
        f"obstacle visible in only {class_frames['obstacle']}/{args.expected} frames; expected >=20%"
    )

if thumbnails:
    columns = 10
    blank = np.zeros_like(thumbnails[0])
    while len(thumbnails) % columns:
        thumbnails.append(blank)
    rows = [
        np.hstack(thumbnails[start : start + columns])
        for start in range(0, len(thumbnails), columns)
    ]
    cv2.imwrite(str(root / "inspection_contact_sheet.jpg"), np.vstack(rows))

report = {
    "passed": not errors,
    "frames_inspected": len(records),
    "errors": errors,
    "class_frame_counts": dict(class_frames),
    "class_pixel_counts": dict(class_pixels),
    "frames": records,
}
(root / "inspection_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {key: report[key] for key in
         ("passed", "frames_inspected", "errors", "class_frame_counts")},
        indent=2,
    )
)
raise SystemExit(0 if not errors else 1)
