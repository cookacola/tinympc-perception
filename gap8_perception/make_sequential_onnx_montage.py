#!/usr/bin/env python3
"""Render a real-flight inference montage from the deployed sequential ONNX."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from gap8_perception.output_contract import OFFSET_MAX, OFFSET_MIN, SCORE_LIMIT

COLORS = ((0, 220, 255), (0, 255, 0), (255, 100, 0), (255, 0, 255))


def integer_score(raw: np.ndarray, scale: float, score_offset: float) -> np.ndarray:
    return raw * scale - score_offset


def run(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    crop = image[20:140, :]
    return session.run(
        None, {session.get_inputs()[0].name: crop.astype(np.float32)[None, None]}
    )[0]


def render(
    image: np.ndarray,
    scores: np.ndarray,
    name: str,
    scale: float,
    score_offset: float,
) -> tuple[np.ndarray, dict]:
    logical = integer_score(scores, scale, score_offset)
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    peaks = []
    for channel, color in enumerate(COLORS):
        heatmap = logical[0, channel]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        u = 160.0 / 20.0 * (float(x) + 0.5) - 0.5
        v = 120.0 / 15.0 * (float(y) + 0.5) - 0.5
        # Image-space output is relative to the 120-row camera crop.
        point = (int(round(u)), int(round(v + 20)))
        cv2.drawMarker(panel, point, color, cv2.MARKER_CROSS, 14, 1)
        peaks.append(float(heatmap[y, x]))
    offset_scores = logical[0, 4:8].mean(axis=(-2, -1))
    confidence = logical[0, 8:12].mean(axis=(-2, -1))
    offsets = OFFSET_MIN + (
        np.clip(offset_scores, -SCORE_LIMIT, SCORE_LIMIT) + SCORE_LIMIT
    ) / (2.0 * SCORE_LIMIT) * (OFFSET_MAX - OFFSET_MIN)
    gate_like = min(peaks) >= 0.0
    label = "GATE-LIKE" if gate_like else "NO GATE (score threshold)"
    cv2.rectangle(panel, (0, 0), (160, 39), (0, 0, 0), -1)
    cv2.putText(panel, name, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (255, 255, 255), 1)
    cv2.putText(panel, label, (3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.29,
                (0, 255, 0) if gate_like else (0, 80, 255), 1)
    cv2.putText(panel, "peak " + "/".join("%.1f" % value for value in peaks),
                (3, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (220, 220, 220), 1)
    cv2.putText(panel, "d " + "/".join("%.1f" % value for value in offsets),
                (3, 151), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    cv2.putText(panel, "c " + "/".join("%.1f" % value for value in confidence),
                (3, 159), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    return panel, {
        "file": name,
        "classification": label,
        "corner_peak_scores": peaks,
        "offset_m": offsets.tolist(),
        "confidence_scores": confidence.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--quantization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.png", help="image glob within --images")
    parser.add_argument("--individual-dir", type=Path,
                        help="also write one annotated PNG per selected image")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--tiles", type=int, default=16)
    args = parser.parse_args()
    files = sorted(args.images.glob(args.pattern))
    if not files:
        raise FileNotFoundError("no PNGs in %s" % args.images)
    manifest = json.loads(args.quantization.read_text())
    scale = float(manifest["scale"])
    score_offset = float(manifest["logical_score_offset"])
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    sampled = [files[index] for index in np.linspace(0, len(files) - 1, min(args.sample_count, len(files))).astype(int)]
    scored = []
    for path in sampled:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            continue
        prediction = run(session, image)
        logical = integer_score(prediction, scale, score_offset)
        peak_floor = float(min(logical[0, channel].max() for channel in range(4)))
        scored.append((peak_floor, path, image, prediction))
    if not scored:
        raise RuntimeError("no usable 160x160 images")
    # Cover the output range rather than cherry-picking only strongest detections.
    ordered = sorted(scored, key=lambda item: item[0])
    selected = [ordered[index] for index in np.linspace(0, len(ordered) - 1, min(args.tiles, len(ordered))).astype(int)]
    panels, records = [], []
    for _, path, image, prediction in selected:
        panel, record = render(image, prediction, path.name, scale, score_offset)
        panels.append(panel)
        records.append(record)
        if args.individual_dir:
            args.individual_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.individual_dir / path.name), panel):
                raise RuntimeError("could not write annotated image %s" % path.name)
    columns = int(np.ceil(np.sqrt(len(panels))))
    rows = int(np.ceil(len(panels) / columns))
    blank = np.zeros_like(panels[0])
    panels += [blank] * (rows * columns - len(panels))
    montage = cv2.vconcat([cv2.hconcat(panels[row * columns:(row + 1) * columns]) for row in range(rows)])
    montage = cv2.resize(montage, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise RuntimeError("could not write montage")
    args.output.with_suffix(".json").write_text(json.dumps({
        "onnx": str(args.onnx), "quantization": manifest,
        "images_dir": str(args.images), "records": records,
        "notes": "Raw deployed-ONNX predictions only; gate labels are not PnP or temporal validation.",
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
