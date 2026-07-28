"""160x120 view of the existing calibrated multitask corpus."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path

from .data import MultiTaskDataset


class STDCMultiTaskDataset(MultiTaskDataset):
    """Center-crop square HM01B0 records to the design document's 4:3 view.

    Cached targets are transformed through their native image domains rather
    than sliced approximately: corner coordinates are shifted by the 20-pixel
    crop, and dense 20x20 danger is expanded, cropped, then area-resampled to
    20x15.
    """

    crop_top = 20
    crop_bottom = 140

    @staticmethod
    def _crop_dense(array: np.ndarray, output_width: int, output_height: int):
        native = cv2.resize(array, (160, 160), interpolation=cv2.INTER_NEAREST)
        cropped = native[20:140, :]
        return cv2.resize(
            cropped,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    @staticmethod
    def _crop_corner_maps(array: np.ndarray) -> np.ndarray:
        cropped = []
        for channel in array:
            native = cv2.resize(channel, (160, 160), interpolation=cv2.INTER_LINEAR)
            cropped.append(
                cv2.resize(
                    native[20:140, :],
                    (40, 30),
                    interpolation=cv2.INTER_AREA,
                )
            )
        return np.asarray(cropped, dtype=np.float32)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        image = batch["image"][:, self.crop_top : self.crop_bottom, :]
        corner_maps = self._crop_corner_maps(batch["corners"].numpy())
        valid = bool(batch["corner_valid"]) and bool(
            (corner_maps.reshape(4, -1).max(axis=1) > 0.25).all()
        )
        batch["image"] = image.contiguous()
        batch["corners"] = torch.from_numpy(
            corner_maps if valid else np.zeros_like(corner_maps)
        )
        batch["corner_valid"] = torch.tensor(valid, dtype=torch.bool)
        batch["danger"] = torch.from_numpy(
            self._crop_dense(batch["danger"][0].numpy(), 20, 15)
        ).unsqueeze(0)
        return batch


class STDCPrivilegedDataset(STDCMultiTaskDataset):
    """Add inverse metric depth for the training-only teacher."""

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        source = Path(batch["source"])
        suffix = source.stem.rsplit("_", 1)[-1]
        depth_path = source.with_name(f"depth_mm_{suffix}.png")
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.shape != (160, 160):
            raise ValueError(f"invalid privileged depth: {depth_path}")
        depth_m = depth_mm[20:140].astype(np.float32) * 0.001
        inverse_depth = np.zeros_like(depth_m)
        valid = depth_m > 0
        inverse_depth[valid] = np.clip(1.0 / depth_m[valid], 0.0, 2.0) / 2.0
        batch["privileged_image"] = torch.cat(
            (batch["image"], torch.from_numpy(inverse_depth).unsqueeze(0)), dim=0
        )
        return batch
