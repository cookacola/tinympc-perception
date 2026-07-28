"""Labeled gate-only real-flight dataset for corner-domain adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .audit_real_flights import canonical_image_order


class RealCornerDataset(Dataset):
    def __init__(self, root: Path, flights: tuple[str, ...]):
        self.records = []
        for flight in flights:
            folder = root / flight
            for line in (folder / "labels.jsonl").read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                corners = canonical_image_order(
                    np.asarray(row["corners"], np.float32).reshape(4, 2)
                )[0]
                if (corners[:, 1] < 20).any() or (corners[:, 1] >= 140).any():
                    continue
                self.records.append(
                    (folder / "stream_out" / row["image"], corners)
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, corners = self.records[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(f"invalid real HM01B0 frame: {path}")
        yy, xx = np.mgrid[:30, :40]
        maps = np.zeros((4, 30, 40), np.float32)
        scaled = corners.copy()
        scaled[:, 0] *= 40.0 / 160.0
        scaled[:, 1] = (scaled[:, 1] - 20.0) * 30.0 / 120.0
        for channel, (x, y) in enumerate(scaled):
            maps[channel] = np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.25**2)
            )
        return {
            "image": torch.from_numpy(image[20:140].copy())
            .unsqueeze(0)
            .float()
            / 255.0,
            "corners": torch.from_numpy(maps),
            "corner_valid": torch.tensor(True),
            "corner_xy": torch.from_numpy(corners.copy()),
            "source": str(path),
        }
