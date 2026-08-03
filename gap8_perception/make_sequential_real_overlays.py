#!/usr/bin/env python3
"""Render predicted gate corners on held-out real-flight frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .model_sequential import SequentialSTDCNet
from .quantization import prepare_int8_qat


COLORS = ((0, 220, 0), (0, 180, 255), (255, 190, 0), (220, 0, 220))


def draw_marker(image: np.ndarray, point: np.ndarray, color, label: str, radius: int) -> None:
    x, y = (int(round(value)) for value in point)
    cv2.circle(image, (x, y), radius, color, 1, cv2.LINE_AA)
    cv2.putText(image, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--flight", default="flight_08")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--montage-columns", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SequentialSTDCNet()
    if state.get("quantization_aware"):
        model = prepare_int8_qat(model)
    model.load_state_dict(state["model"])
    model.eval()

    flight_dir = args.real_root / args.flight
    rows = [json.loads(line) for line in (flight_dir / "labels.jsonl").read_text().splitlines() if line]
    if not rows:
        raise RuntimeError("no held-out flight frames")
    indices = np.linspace(0, len(rows) - 1, min(args.samples, len(rows)), dtype=int)
    selected = [rows[index] for index in indices]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest, tiles = [], []
    for start in range(0, len(selected), args.batch_size):
        chunk = selected[start:start + args.batch_size]
        frames = []
        for row in chunk:
            image = cv2.imread(str(flight_dir / "stream_out" / row["image"]), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (160, 160):
                raise ValueError(f"invalid image {row['image']}")
            frames.append(image)
        with torch.no_grad():
            output = model(torch.from_numpy(np.asarray(frames)[:, 20:140]).unsqueeze(1).float() / 255.0)
        fields = output[:, :4].numpy()
        for local, (row, frame, field) in enumerate(zip(chunk, frames, fields)):
            prediction, scores = [], []
            for channel in field:
                y, x = np.unravel_index(np.argmax(channel), channel.shape)
                prediction.append((8.0 * (x + 0.5) - 0.5, 8.0 * (y + 0.5) + 19.5))
                scores.append(float(channel.max()))
            prediction = np.asarray(prediction, np.float32)
            canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for corner, color in enumerate(COLORS):
                draw_marker(canvas, prediction[corner], color, f"P{corner}", 2)
            cv2.rectangle(canvas, (0, 0), (159, 15), (0, 0, 0), -1)
            cv2.putText(canvas, "QAT predicted corners", (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            filename = f"{start + local:03d}_{Path(row['image']).stem}.png"
            cv2.imwrite(str(args.output / filename), canvas)
            manifest.append({"file": filename, "source": row["image"], "corner_scores": scores})
            tiles.append(canvas)
    columns = max(1, args.montage_columns)
    rows_needed = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (rows_needed * columns - len(tiles))
    montage = np.vstack([
        np.hstack(padded[row * columns:(row + 1) * columns])
        for row in range(rows_needed)
    ])
    cv2.imwrite(str(args.output / "montage_predictions_only.png"), montage)
    (args.output / "README.txt").write_text(
        "Held-out flight_08 QAT predictions only. P0..P3 mark the four predicted gate corners. montage_predictions_only.png combines all samples.\n"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
