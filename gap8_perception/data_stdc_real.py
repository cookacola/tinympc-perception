"""Labeled gate-only real-flight dataset for corner-domain adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .audit_real_flights import canonical_image_order


def augment_real_gate_frame(
    image: np.ndarray,
    corners: np.ndarray,
    rng: np.random.Generator,
    strength: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply label-preserving pose/exposure variation to one HM01B0 frame."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("augmentation strength must be in [0, 1]")
    original_image = image
    original_corners = corners
    height, width = image.shape
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    for _ in range(4):
        angle = float(rng.uniform(-5.0, 5.0) * strength)
        scale = float(1.0 + rng.uniform(-0.10, 0.10) * strength)
        matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
        matrix[:, 2] += np.asarray(
            [
                rng.uniform(-8.0, 8.0) * strength,
                rng.uniform(-5.0, 5.0) * strength,
            ],
            np.float32,
        )
        homogeneous = np.concatenate(
            [original_corners, np.ones((4, 1), np.float32)], axis=1
        )
        transformed = homogeneous @ matrix.T
        if (
            (transformed[:, 0] >= 1.0).all()
            and (transformed[:, 0] < width - 1.0).all()
            and (transformed[:, 1] >= 21.0).all()
            and (transformed[:, 1] < 139.0).all()
        ):
            image = cv2.warpAffine(
                original_image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            corners = transformed.astype(np.float32)
            break
    else:
        image = original_image.copy()
        corners = original_corners.copy()

    if rng.random() < 0.5 * strength:
        image = np.ascontiguousarray(image[:, ::-1])
        corners[:, 0] = (width - 1) - corners[:, 0]
    corners = canonical_image_order(corners)[0]

    normalized = image.astype(np.float32) / 255.0
    gamma = float(1.0 + rng.uniform(-0.35, 0.55) * strength)
    gain = float(1.0 + rng.uniform(-0.28, 0.28) * strength)
    bias = float(rng.uniform(-0.10, 0.10) * strength)
    normalized = gain * np.power(normalized, gamma) + bias
    normalized += rng.normal(
        0.0, rng.uniform(0.0, 0.025) * strength, image.shape
    )
    if rng.random() < 0.20 * strength:
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0.0)
    image = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    return image, corners


class RealCornerDataset(Dataset):
    def __init__(
        self,
        root: Path,
        flights: tuple[str, ...],
        augment: bool = False,
        augmentation_strength: float = 0.35,
        augmentation_probability: float = 0.0,
    ):
        self.augment = augment
        self.augmentation_strength = augmentation_strength
        self.augmentation_probability = augmentation_probability
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
        if self.augment and torch.rand(()).item() < self.augmentation_probability:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
            image, corners = augment_real_gate_frame(
                image,
                corners,
                np.random.default_rng(seed),
                strength=self.augmentation_strength,
            )
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
