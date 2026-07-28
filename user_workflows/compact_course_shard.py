#!/usr/bin/env python3
"""Encode depth and an aligned HM01B0-style 8-bit monochrome stream."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=Path, required=True)
parser.add_argument("--keep-float-depth", action="store_true")
args = parser.parse_args()

report = json.loads((args.shard / "validation.json").read_text())
if not report.get("passed"):
    raise RuntimeError("refusing to compact an unvalidated shard")

depth_files = sorted(args.shard.glob("distance_to_image_plane_*.npy"))
rgb_files = sorted(args.shard.glob("rgb_*.png"))
if len(rgb_files) != len(depth_files):
    raise RuntimeError("RGB/depth count mismatch before sensor encoding")

for source in rgb_files:
    rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read {source}")
    if rgb.shape[:2] != (160, 160):
        raise RuntimeError(
            f"HM01B0 training output must be 160x160, got {rgb.shape[:2]} in {source}"
        )
    mono = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    destination = source.with_name(source.name.replace("rgb_", "hm01b0_mono_"))
    if not cv2.imwrite(str(destination), mono):
        raise RuntimeError(f"failed to write {destination}")
    check = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if check is None or check.dtype != np.uint8 or check.shape != mono.shape:
        raise RuntimeError(f"failed to verify {destination}")

for source in depth_files:
    depth_m = np.load(source, allow_pickle=False)
    finite = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= 65.535)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[finite] = np.rint(depth_m[finite] * 1000.0).astype(np.uint16)
    destination = source.with_name(source.name.replace("distance_to_image_plane_", "depth_mm_")).with_suffix(".png")
    if not cv2.imwrite(str(destination), depth_mm):
        raise RuntimeError(f"failed to write {destination}")
    check = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if check is None or check.dtype != np.uint16 or check.shape != depth_mm.shape:
        raise RuntimeError(f"failed to verify {destination}")
    if not args.keep_float_depth:
        source.unlink()

(args.shard / "depth_encoding.json").write_text(
    json.dumps(
        {
            "format": "uint16_png",
            "unit": "millimeter",
            "meters_per_unit": 0.001,
            "invalid_value": 0,
            "saturation_m": 65.535,
            "out_of_range_policy": "values above 65.535 m are encoded as invalid 0",
            "source": "Isaac Sim distance_to_image_plane",
        },
        indent=2,
    )
)
(args.shard / "camera_sensor.json").write_text(
    json.dumps(
        {
            "model": "Himax HM01B0",
            "stream": "hm01b0_mono_####.png",
            "modality": "monochrome",
            "encoding": "uint8_png",
            "resolution": [160, 160],
            "readout_mode": "2x2 monochrome binning from 320x320",
            "shutter": "global",
            "alignment": "pixel-aligned with RGB, depth, and semantic segmentation",
            "rgb_note": "RGB is retained as synthetic privileged training data; the deployed sensor input is the monochrome stream.",
            "reference": "https://www.himax.com.tw/products/cmos-image-sensor/always-on-vision-sensors/hm01b0/",
        },
        indent=2,
    )
    + "\n"
)
print(
    f"encoded {len(rgb_files)} HM01B0 monochrome frames and "
    f"compacted {len(depth_files)} aligned metric-depth frames"
)
