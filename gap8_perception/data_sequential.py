"""Datasets for the canonical sequential 12-channel perception contract."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .hm01b0_sensor import augment_hm01b0
from .output_contract import GRID_HEIGHT, GRID_WIDTH, OFFSET_MAX, OFFSET_MIN, SCORE_LIMIT
from .targets import reliable_gate_corner_view


TARGET_SCHEMA = "sequential_fixed_normal_v1"


def corner_score_fields(
    corners_xy160: np.ndarray,
    visibility: bool | np.ndarray,
    sigma: float = 0.85,
) -> np.ndarray:
    """Render ordered 20x15 fields with independent corner visibility."""
    fields = np.full((4, GRID_HEIGHT, GRID_WIDTH), -SCORE_LIMIT, np.float32)
    corners = np.asarray(corners_xy160, np.float32).copy()
    visible = np.broadcast_to(np.asarray(visibility, bool), (4,))
    corners[:, 1] -= 20.0
    if not np.isfinite(corners).all():
        return fields
    yy, xx = np.mgrid[:GRID_HEIGHT, :GRID_WIDTH]
    for channel, (u, v) in enumerate(corners):
        if not visible[channel]:
            continue
        uh = (u + 0.5) / 8.0 - 0.5
        vh = (v + 0.5) / 8.0 - 0.5
        fields[channel] = 2.0 * SCORE_LIMIT * np.exp(
            -((xx - uh) ** 2 + (yy - vh) ** 2) / (2.0 * sigma**2)
        ) - SCORE_LIMIT
    return fields


def encode_offsets(offsets_m: np.ndarray) -> np.ndarray:
    offsets = np.clip(np.asarray(offsets_m, np.float32), OFFSET_MIN, OFFSET_MAX)
    return 2.0 * SCORE_LIMIT * (offsets - OFFSET_MIN) / (
        OFFSET_MAX - OFFSET_MIN
    ) - SCORE_LIMIT


class SequentialTargetDataset(Dataset):
    """Load only canonical fixed-normal geometry targets."""

    crop_top, crop_bottom = 20, 140

    def __init__(
        self,
        dataset_root: Path,
        targets_root: Path,
        split_file: Path,
        split: str,
        limit: int | None = None,
        observation_dropout_probability: float = 0.0,
        sensor_augmentation_probability: float = 0.0,
    ):
        if not 0.0 <= observation_dropout_probability < 1.0:
            raise ValueError("observation_dropout_probability must be in [0, 1)")
        split_data = json.loads(Path(split_file).read_text())
        split_spec = split_data[split]
        if isinstance(split_spec, dict):
            required = {"start", "stop", "step"}
            if set(split_spec) != required:
                raise ValueError(
                    f"invalid range split for {split!r}: expected {sorted(required)}"
                )
            shard_names = [
                f"shard_{start:09d}"
                for start in range(
                    int(split_spec["start"]),
                    int(split_spec["stop"]),
                    int(split_spec["step"]),
                )
            ]
        elif isinstance(split_spec, list):
            shard_names = split_spec
        else:
            raise TypeError(f"unsupported split specification for {split!r}")

        self.records = []
        for shard_name in shard_names:
            shard = Path(dataset_root) / shard_name
            target = Path(targets_root) / f"{shard_name}.npz"
            if not target.is_file():
                raise FileNotFoundError(target)
            for local, image in enumerate(sorted(shard.glob("hm01b0_mono_*.png"))):
                if limit is not None and len(self.records) >= limit:
                    break
                self.records.append((shard_name, local, image, target))
            if limit is not None and len(self.records) >= limit:
                break
        self.observation_dropout_probability = observation_dropout_probability
        self.sensor_augmentation_probability = sensor_augmentation_probability
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.records)

    def _targets(self, shard_name: str, path: Path) -> dict[str, np.ndarray]:
        if shard_name not in self._cache:
            with np.load(path) as archive:
                required = {
                    "schema",
                    "corners_xy160_f16",
                    "corner_visibility_u8",
                    "fixed_normal_offsets_m_f16",
                    "fixed_normal_confidence_u8",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise RuntimeError(
                        f"{path} is not canonical; missing {sorted(missing)}"
                    )
                schema = str(archive["schema"].item())
                if schema != TARGET_SCHEMA:
                    raise RuntimeError(f"unsupported target schema {schema!r} in {path}")
                self._cache[shard_name] = {
                    key: archive[key] for key in archive.files
                }
        return self._cache[shard_name]

    def __getitem__(self, index):
        shard_name, local, image_path, target_path = self.records[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(f"invalid HM01B0 image: {image_path}")
        target = self._targets(shard_name, target_path)
        corners = target["corners_xy160_f16"][local].astype(np.float32)
        visibility = target["corner_visibility_u8"][local].astype(bool)
        visibility &= reliable_gate_corner_view(corners)
        offset_m = target["fixed_normal_offsets_m_f16"][local].astype(np.float32)
        offset_valid = (
            target["fixed_normal_confidence_u8"][local].astype(np.float32) / 255.0
        )
        packed = np.empty((12, GRID_HEIGHT, GRID_WIDTH), np.float32)
        packed[:4] = corner_score_fields(corners, visibility)
        packed[4:8] = encode_offsets(offset_m)[:, None, None]
        packed[8:12] = np.where(
            offset_valid[:, None, None] > 0.5, SCORE_LIMIT, -SCORE_LIMIT
        )
        offset_loss_mask = np.ones(4, np.float32)
        confidence_loss_mask = np.ones(4, np.float32)

        image, observable = augment_hm01b0(
            image, np.random.default_rng(), self.sensor_augmentation_probability
        )
        if not observable or np.random.random() < self.observation_dropout_probability:
            if observable:
                image = np.zeros_like(image)
            packed[:4] = -SCORE_LIMIT
            packed[8:12] = -SCORE_LIMIT
            visibility[:] = False
            offset_valid[:] = 0.0
            offset_loss_mask[:] = 0.0

        return {
            "image": torch.from_numpy(
                image[self.crop_top:self.crop_bottom].copy()
            ).unsqueeze(0).float() / 255.0,
            "target": torch.from_numpy(packed),
            "corner_valid": torch.tensor(bool(visibility.all()), dtype=torch.bool),
            "corner_visibility": torch.from_numpy(visibility.copy()),
            "corner_xy_crop": torch.from_numpy(
                np.stack((corners[:, 0], corners[:, 1] - self.crop_top), axis=1)
            ),
            "offset_m": torch.from_numpy(offset_m),
            "offset_valid": torch.from_numpy(offset_valid),
            "offset_loss_mask": torch.from_numpy(offset_loss_mask),
            "confidence_loss_mask": torch.from_numpy(confidence_loss_mask),
            "source": str(image_path),
        }


class RealSequentialCornerDataset(Dataset):
    """Real centerline-corner labels without invented safety supervision."""

    crop_top, crop_bottom = 20, 140

    def __init__(
        self,
        root: Path,
        flights: tuple[str, ...],
        sensor_augmentation_probability: float = 0.0,
    ):
        self.sensor_augmentation_probability = sensor_augmentation_probability
        self.records = []
        for flight in flights:
            labels = Path(root) / flight / "labels.jsonl"
            if not labels.is_file():
                raise FileNotFoundError(labels)
            for line in labels.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                corners = np.asarray(row["corners"], np.float32).reshape(4, 2)
                visibility = (
                    (corners[:, 0] >= 0)
                    & (corners[:, 0] < 160)
                    & (corners[:, 1] >= self.crop_top)
                    & (corners[:, 1] < self.crop_bottom)
                )
                if visibility.any():
                    self.records.append(
                        (labels.parent / "stream_out" / row["image"], corners, visibility)
                    )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, corners, visibility = self.records[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(f"invalid real HM01B0 image: {path}")
        image, observable = augment_hm01b0(
            image, np.random.default_rng(), self.sensor_augmentation_probability
        )
        visibility = visibility & observable
        packed = np.full((12, GRID_HEIGHT, GRID_WIDTH), -SCORE_LIMIT, np.float32)
        packed[:4] = corner_score_fields(corners, visibility)
        return {
            "image": torch.from_numpy(
                image[self.crop_top:self.crop_bottom].copy()
            ).unsqueeze(0).float() / 255.0,
            "target": torch.from_numpy(packed),
            "corner_valid": torch.tensor(bool(visibility.all()), dtype=torch.bool),
            "corner_visibility": torch.from_numpy(visibility.copy()),
            "corner_xy_crop": torch.from_numpy(
                np.stack((corners[:, 0], corners[:, 1] - self.crop_top), axis=1)
            ),
            "offset_m": torch.zeros(4),
            "offset_valid": torch.zeros(4),
            "offset_loss_mask": torch.zeros(4),
            "confidence_loss_mask": torch.zeros(4),
            "source": str(path),
        }
