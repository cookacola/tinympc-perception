#!/usr/bin/env python3
"""Run the deployed sequential QAT ONNX model on a grayscale image using CPU ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
GRID_WIDTH, GRID_HEIGHT = 20, 15
SCORE_LIMIT = 6.0
OFFSET_MAX = 6.0
COLORS = ((0, 220, 255), (0, 255, 0), (255, 100, 0), (255, 0, 255))
DIRECTION_DEGREES = (-40.0, -13.333333, 13.333333, 40.0)
DIRECTION_NAMES = ("outer", "center", "center", "outer")


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's sensor-frame/crop contract to a monochrome image."""
    if image.shape == (160, 160):
        sensor_frame = image
        crop = image[20:140, :]
    elif image.shape == (120, 160):
        crop = image
        sensor_frame = np.pad(image, ((20, 20), (0, 0)), mode="edge")
    else:
        raise ValueError("expected a grayscale 160x160 sensor frame or 160x120 center crop")
    return sensor_frame, crop.astype(np.float32)[None, None]


def preprocess(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read {path}")
    return preprocess_image(image)


def decode(raw: np.ndarray, scale: float) -> dict:
    if raw.shape != (1, 12, GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"unexpected ONNX output shape {raw.shape}")
    logical = raw.astype(np.float32)[0] * scale - SCORE_LIMIT
    corners, peaks, ambiguity = [], [], []
    for heatmap in logical[:4]:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        competing = heatmap.copy()
        competing[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -np.inf
        corners.append([8.0 * (float(x) + 0.5) - 0.5, 8.0 * (float(y) + 0.5) - 0.5])
        peaks.append(float(heatmap[y, x]))
        ambiguity.append(float(heatmap[y, x] - np.max(competing)))
    offset_scores = logical[4:8].mean(axis=(1, 2))
    offsets = (np.clip(offset_scores, -SCORE_LIMIT, SCORE_LIMIT) + SCORE_LIMIT) / (2 * SCORE_LIMIT) * OFFSET_MAX
    confidence = logical[8:12].mean(axis=(1, 2))
    contour = np.rint(np.asarray(corners, np.float32)).astype(np.int32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(np.asarray(corners, np.float32))))
    sides = np.linalg.norm(np.asarray(corners) - np.roll(np.asarray(corners), -1, axis=0), axis=1)
    gate_valid = bool(
        min(peaks) >= 0.0 and min(ambiguity) >= 0.5 and cv2.isContourConvex(contour)
        and area >= 100.0 and sides.min() > 0 and sides.max() / sides.min() <= 6.0
    )
    return {
        "gate_valid": gate_valid,
        "corners_xy_crop": corners,
        "corner_peak_scores": peaks,
        "corner_ambiguity_margins": ambiguity,
        "clearance_m": offsets.tolist(),
        "clearance_confidence_scores": confidence.tolist(),
    }


def _text(panel: np.ndarray, text: str, origin: tuple[int, int], scale: float,
          color: tuple[int, int, int] = (235, 235, 235), thickness: int = 1) -> None:
    cv2.putText(panel, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _direction_panel(result: dict, safe_min: float, confidence_min: float) -> np.ndarray:
    """Render all four directional outputs outside the camera image."""
    panel = np.full((208, 270, 3), (22, 22, 22), dtype=np.uint8)
    _text(panel, "Directional clearance classifier", (10, 18), 0.44)
    _text(panel, f"open: d >= {safe_min:.2f} m or c < {confidence_min:+.2f}",
          (10, 34), 0.33, (180, 180, 180))
    clearance = result["clearance_m"]
    confidence = result["clearance_confidence_scores"]
    for index, (degrees, name, distance, score) in enumerate(
            zip(DIRECTION_DEGREES, DIRECTION_NAMES, clearance, confidence)):
        top = 43 + 32 * index
        open_direction = score < confidence_min or distance >= safe_min
        color = (55, 190, 55) if open_direction else (40, 70, 235)
        cv2.rectangle(panel, (7, top), (263, top + 28), (38, 38, 38), -1)
        cv2.rectangle(panel, (7, top), (263, top + 28), color, 1)
        _text(panel, f"{degrees:+05.1f}  {name}", (13, top + 11), 0.34, color)
        state = "OPEN" if open_direction else "BLOCKED"
        _text(panel, f"{distance:.2f} m  c {score:+.2f}  {state}",
              (13, top + 24), 0.34, (235, 235, 235))
    _text(panel, "Bits: 0=-40  1=-13.3  2=+13.3  3=+40", (10, 197),
          0.31, (180, 180, 180))
    return panel


def annotate(sensor_frame: np.ndarray, result: dict, safe_min: float = 0.32,
             confidence_min: float = 0.0) -> np.ndarray:
    """Show the complete camera view beside all four direction decisions."""
    camera = cv2.cvtColor(sensor_frame, cv2.COLOR_GRAY2BGR)
    for point, color in zip(result["corners_xy_crop"], COLORS):
        cv2.drawMarker(camera, (round(point[0]), round(point[1] + 20)), color,
                       cv2.MARKER_CROSS, 14, 1)
    state = "GATE VALID" if result["gate_valid"] else "GATE REJECTED"
    camera_panel = np.full((208, 160, 3), (22, 22, 22), dtype=np.uint8)
    camera_panel[:160] = camera
    _text(camera_panel, state, (4, 181), 0.43,
          (0, 255, 0) if result["gate_valid"] else (0, 80, 255))
    _text(camera_panel, "full 160x160 sensor frame", (4, 199), 0.31,
          (180, 180, 180))
    return np.hstack((camera_panel,
                      _direction_panel(result, safe_min, confidence_min)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "sequential_int.onnx")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--annotated-image", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "quantization_manifest.json").read_text())
    frame, tensor = preprocess(args.image)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    result = decode(raw, float(manifest["scale"]))
    result["model_sha256"] = "620fdb49f94abd7adf212b15b0858c49ed46f85f89fdbc4e05d28453c5c9f9b6"
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.annotated_image:
        if not cv2.imwrite(str(args.annotated_image), annotate(frame, result)):
            raise RuntimeError(f"could not write {args.annotated_image}")


if __name__ == "__main__":
    main()
