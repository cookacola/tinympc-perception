"""Dataset loader for aligned HM01B0 images and cached multi-task targets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import gaussian_heatmaps, reliable_gate_corner_view


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        targets_root: Path,
        split_file: Path,
        split: str,
        limit: int | None = None,
        augment: bool = False,
        high_speed_stride: int = 1,
        expand_state_variants: bool = False,
    ):
        split_data = json.loads(split_file.read_text())
        self.records = []
        images_seen = 0
        for shard_name in split_data[split]:
            shard = dataset_root / shard_name
            target_path = targets_root / f"{shard_name}.npz"
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            with np.load(target_path) as archive:
                variants = int(archive["vehicle_state_f32"].shape[1])
                nominal_variant = int(np.argmin(np.abs(
                    archive["speed_variants_mps"] - 1.0
                )))
            for local, mono in enumerate(sorted(shard.glob("hm01b0_mono_*.png"))):
                if limit is not None and images_seen >= limit:
                    break
                selected_variants = (
                    range(variants)
                    if expand_state_variants
                    else (nominal_variant,)
                )
                for variant in selected_variants:
                    if (
                        variant == variants - 1
                        and high_speed_stride > 1
                        and local % high_speed_stride != 0
                    ):
                        continue
                    self.records.append(
                        (shard_name, local, variant, mono, target_path)
                    )
                images_seen += 1
            if limit is not None and images_seen >= limit:
                break
        self.augment = augment
        self.expand_state_variants = expand_state_variants
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.records)

    def _targets(self, shard_name: str, path: Path):
        if shard_name not in self._cache:
            archive = np.load(path)
            self._cache[shard_name] = {key: archive[key] for key in archive.files}
            archive.close()
        return self._cache[shard_name]

    def __getitem__(self, index):
        shard_name, local, variant, mono_path, target_path = self.records[index]
        mono = cv2.imread(str(mono_path), cv2.IMREAD_GRAYSCALE)
        if mono is None or mono.shape != (160, 160):
            raise ValueError(f"invalid HM01B0 image: {mono_path}")
        target = self._targets(shard_name, target_path)
        corners = target["corners_xy160_f16"][local].astype(np.float32)
        # Apply the current visibility/conditioning rule even to older target
        # caches so distant and nearly edge-on gates never supervise corners.
        valid = bool(target["corner_valid_u8"][local]) and reliable_gate_corner_view(
            corners
        )
        corner_maps = gaussian_heatmaps(corners, valid)
        # A single-input GAP8 network cannot learn speed-dependent risk. For
        # normal training use the exact nominal-speed horizon rollout; the
        # controller combines its range output with live speed and latency.
        # ``obstacle_presence_u8`` means any geometry within 6 m and is too
        # broad to serve as a collision target.
        danger_key = "danger_u8"
        danger_source = target[danger_key][local]
        if danger_key == "danger_u8":
            danger_source = danger_source[variant]
        danger = danger_source.astype(np.float32) / 255.0
        # rendercal-v2 stores the same control-oriented auxiliary supervision
        # under its original ``urgency_u8`` name.  Calibrated-v3 additionally
        # exposes it as ``inverse_range_u8``.
        urgency_key = (
            "inverse_range_u8"
            if "inverse_range_u8" in target
            else "urgency_u8"
        )
        urgency_source = target[urgency_key][local]
        if urgency_source.ndim == 3:
            urgency_source = urgency_source[variant]
        urgency = urgency_source.astype(np.float32) / 255.0
        uncertainty = (
            target["uncertainty_u8"][local, variant].astype(np.float32) / 255.0
        )
        clearance = target["minimum_clearance_m_f16"][local, variant].astype(
            np.float32
        )
        ttc = target["time_to_collision_s_f16"][local, variant].astype(
            np.float32
        )
        vehicle_state = target["vehicle_state_f32"][local, variant].astype(
            np.float32
        )
        gate = target["gate_opening_u8"][local].astype(np.float32)
        if self.augment:
            (
                mono,
                corner_maps,
                danger,
                urgency,
                uncertainty,
                gate,
                vehicle_state,
            ) = self._augment(
                mono,
                corner_maps,
                danger,
                urgency,
                uncertainty,
                gate,
                vehicle_state,
            )
        return {
            "image": torch.from_numpy(mono.copy()).unsqueeze(0).float() / 255.0,
            "corners": torch.from_numpy(corner_maps.copy()),
            "corner_xy": torch.from_numpy(corners),
            "corner_valid": torch.tensor(valid, dtype=torch.bool),
            "danger": torch.from_numpy(danger.copy()).unsqueeze(0),
            "urgency": torch.from_numpy(urgency.copy()).unsqueeze(0),
            "uncertainty": torch.from_numpy(uncertainty.copy()).unsqueeze(0),
            "minimum_clearance_m": torch.from_numpy(clearance.copy()).unsqueeze(0),
            "time_to_collision_s": torch.from_numpy(ttc.copy()).unsqueeze(0),
            "vehicle_state": torch.from_numpy(vehicle_state.copy()),
            "gate": torch.from_numpy(gate.copy()).unsqueeze(0),
            "global_index": int(target["global_indices_i32"][local]),
            "state_variant": variant,
            "gate_index": int(target["gate_index_i8"][local]),
            "gate_projected_corners": torch.from_numpy(
                target["gate_projected_corners_xy160_f16"][local].astype(
                    np.float32
                )
            ),
            "gate_object_corners_m": torch.from_numpy(
                target["gate_object_corners_m_f32"][local].astype(np.float32)
            ),
            "gate_projection_error_px": torch.tensor(
                float(target["gate_projection_error_px_f16"][local]),
                dtype=torch.float32,
            ),
            "gate_translation_camera_m": torch.from_numpy(
                target["gate_translation_camera_m_f32"][local].astype(
                    np.float32
                )
            ),
            "gate_rotation_camera": torch.from_numpy(
                target["gate_rotation_camera_f32"][local].astype(np.float32)
            ),
            "source": str(mono_path),
        }

    @staticmethod
    def _augment(
        mono, corners, danger, urgency, uncertainty, gate, vehicle_state
    ):
        rng = np.random.default_rng()
        # Modest projective jitter, applied consistently at every target scale.
        if rng.random() < 0.5:
            jitter = rng.uniform(-5.0, 5.0, (4, 2)).astype(np.float32)
            source = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
            matrix = cv2.getPerspectiveTransform(source, source + jitter)
            mono = cv2.warpPerspective(mono, matrix, (160, 160), borderMode=cv2.BORDER_REFLECT)
            for array, size, interpolation in (
                (corners, 40, cv2.INTER_LINEAR),
                (danger[None], 20, cv2.INTER_LINEAR),
                (urgency[None], 20, cv2.INTER_LINEAR),
                (uncertainty[None], 20, cv2.INTER_LINEAR),
                (gate[None], 40, cv2.INTER_NEAREST),
            ):
                scaled = matrix.copy()
                scale = size / 160.0
                scale_matrix = np.diag([scale, scale, 1.0])
                scaled = scale_matrix @ matrix @ np.linalg.inv(scale_matrix)
                for channel in range(array.shape[0]):
                    array[channel] = cv2.warpPerspective(
                        array[channel], scaled, (size, size),
                        flags=interpolation, borderMode=cv2.BORDER_CONSTANT,
                    )
        if rng.random() < 0.5:
            mono = mono[:, ::-1]
            corners = corners[[1, 0, 3, 2], :, ::-1]
            danger = danger[:, ::-1]
            urgency = urgency[:, ::-1]
            uncertainty = uncertainty[:, ::-1]
            gate = gate[:, ::-1]
            vehicle_state = vehicle_state.copy()
            vehicle_state[1] *= -1  # body lateral velocity
            vehicle_state[3] *= -1  # axial-vector reflection
            vehicle_state[5] *= -1  # yaw rate
        image = mono.astype(np.float32) / 255.0
        image = np.clip(image * rng.uniform(0.75, 1.25), 0, 1)
        image = np.power(image, rng.uniform(0.75, 1.35))
        # Gate-fabric intensity/texture variation. The cached target is the
        # opening, so an exterior ring provides a conservative frame region.
        opening160 = cv2.resize(gate, (160, 160), interpolation=cv2.INTER_NEAREST)
        if opening160.any() and rng.random() < 0.6:
            outer = cv2.dilate(
                opening160,
                cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
            )
            frame_region = (outer > 0) & (opening160 == 0)
            texture = cv2.resize(
                rng.uniform(0.75, 1.25, (20, 20)).astype(np.float32),
                (160, 160),
                interpolation=cv2.INTER_CUBIC,
            )
            image[frame_region] *= texture[frame_region] * rng.uniform(0.65, 1.35)
        # Low-frequency background illumination randomization outside the gate.
        if rng.random() < 0.5:
            protected = cv2.dilate(
                (opening160 > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            )
            background = protected == 0
            field = cv2.resize(
                rng.uniform(0.7, 1.3, (8, 8)).astype(np.float32),
                (160, 160),
                interpolation=cv2.INTER_CUBIC,
            )
            image[background] *= field[background]
        if rng.random() < 0.5:
            image += rng.normal(0, rng.uniform(0.005, 0.04), image.shape)
        if rng.random() < 0.25:
            kernel = int(rng.choice([3, 5]))
            image = cv2.GaussianBlur(image, (kernel, kernel), rng.uniform(0.3, 1.2))
        if rng.random() < 0.20:
            kernel = np.zeros((5, 5), np.float32)
            if rng.random() < 0.5:
                kernel[2, :] = 0.2
            else:
                kernel[:, 2] = 0.2
            image = cv2.filter2D(image, -1, kernel)
        if rng.random() < 0.25:
            x0 = int(rng.integers(0, 120))
            y0 = int(rng.integers(0, 120))
            image[y0 : y0 + int(rng.integers(10, 45)), x0 : x0 + int(rng.integers(10, 45))] *= rng.uniform(0.2, 0.7)
        return (
            np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8),
            np.ascontiguousarray(corners),
            np.ascontiguousarray(danger),
            np.ascontiguousarray(urgency),
            np.ascontiguousarray(uncertainty),
            np.ascontiguousarray(gate),
            np.ascontiguousarray(vehicle_state),
        )
