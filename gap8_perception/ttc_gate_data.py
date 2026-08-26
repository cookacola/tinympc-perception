"""Layout-disjoint two-frame dataset for joint TTC and gate supervision."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def rotation_from_rpy(rpy):
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray((
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    ), dtype=np.float64)


class TTCGateDataset(Dataset):
    corner_order = ("TL", "TR", "BR", "BL")

    DEFAULT_MAXIMUM_GATE_DISTANCE_M = 8.0
    DEFAULT_MINIMUM_GATE_SPAN_PX = 16.0
    DEFAULT_MINIMUM_GATE_AREA_PX2 = 256.0

    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        maximum_gate_distance_m: float = DEFAULT_MAXIMUM_GATE_DISTANCE_M,
        minimum_gate_span_px: float = DEFAULT_MINIMUM_GATE_SPAN_PX,
        minimum_gate_area_px2: float = DEFAULT_MINIMUM_GATE_AREA_PX2,
    ):
        self.root = Path(root)
        self.augment = bool(augment)
        self.maximum_gate_distance_m = float(maximum_gate_distance_m)
        self.minimum_gate_span_px = float(minimum_gate_span_px)
        self.minimum_gate_area_px2 = float(minimum_gate_area_px2)
        self.trajectories, self.samples = [], []
        dataset = json.loads((self.root / "dataset_manifest.json").read_text())
        for layout in dataset["layouts"]:
            if layout["split"] != split:
                continue
            layout_dir = self.root / layout.get("layout_path", layout["layout_id"])
            scene = json.loads((layout_dir / layout["scene_geometry"]).read_text())
            for summary in layout["trajectories"]:
                trajectory_dir = layout_dir / "trajectories" / summary["trajectory_id"]
                target_path = trajectory_dir / "ttc_targets_20x20.npz"
                if not target_path.exists():
                    raise FileNotFoundError(f"{target_path} is missing")
                frames = [
                    json.loads(line)
                    for line in (trajectory_dir / "frames.jsonl").read_text().splitlines()
                ]
                with np.load(target_path, allow_pickle=False) as archive:
                    targets = {key: archive[key] for key in archive.files}
                if len(frames) != len(targets["frame_indices_i32"]):
                    raise ValueError(f"frame/target mismatch in {trajectory_dir}")
                gate_geometry = self._gate_geometry(frames, targets, scene["gates"])
                trajectory_index = len(self.trajectories)
                trajectory_type = summary.get(
                    "requested_trajectory_type", summary["trajectory_type"]
                )
                self.trajectories.append({
                    "layout_id": layout["layout_id"],
                    "scenario": scene["scenario"],
                    "trajectory_id": summary["trajectory_id"],
                    "trajectory_type": trajectory_type,
                    "dir": trajectory_dir,
                    "frames": frames,
                    "targets": targets,
                    "gate_geometry": gate_geometry,
                })
                self.samples.extend((trajectory_index, current) for current in range(1, len(frames)))

    def __len__(self):
        return len(self.samples)

    def _gate_geometry(self, frames, targets, gates):
        """Compute an independent quality mask without changing visibility labels."""
        gate_indices = targets["gate_index_i16"].astype(np.int64)
        corners = targets["gate_corners_px_f32"].astype(np.float64)
        visible = targets["gate_corners_valid_u8"].astype(bool)
        count = len(gate_indices)
        distance = np.full(count, -1.0, dtype=np.float32)
        selected = (gate_indices >= 0) & (gate_indices < len(gates))
        for index in np.flatnonzero(selected):
            gate_center = np.asarray(gates[gate_indices[index]]["center_m"], dtype=np.float64)
            vehicle_position = np.asarray(
                frames[index]["vehicle_state"]["position_m"], dtype=np.float64
            )
            distance[index] = np.linalg.norm(gate_center - vehicle_position)

        top = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
        right = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
        bottom = np.linalg.norm(corners[:, 3] - corners[:, 2], axis=1)
        left = np.linalg.norm(corners[:, 0] - corners[:, 3], axis=1)
        width = (top + bottom) * 0.5
        height = (left + right) * 0.5
        x, y = corners[:, :, 0], corners[:, :, 1]
        area = 0.5 * np.abs(
            (x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1)
        )
        finite_geometry = (
            np.isfinite(corners).all(axis=(1, 2))
            & np.isfinite(distance)
            & np.isfinite(width)
            & np.isfinite(height)
            & np.isfinite(area)
        )
        resolved_selected_gate = (
            selected
            & visible.any(axis=1)
            & finite_geometry
            & (distance <= self.maximum_gate_distance_m)
            & (width >= self.minimum_gate_span_px)
            & (height >= self.minimum_gate_span_px)
            & (area >= self.minimum_gate_area_px2)
        )
        # A frame with no selected forward gate remains a useful, true negative.
        eligible = (~selected) | resolved_selected_gate
        return {
            "selected": selected,
            "eligible": eligible,
            "distance_m": distance,
            "width_px": np.minimum(width, np.finfo(np.float32).max).astype(np.float32),
            "height_px": np.minimum(height, np.finfo(np.float32).max).astype(np.float32),
            "area_px2": np.minimum(area, np.finfo(np.float32).max).astype(np.float32),
        }

    def gate_supervision_policy(self):
        return {
            "semantics": (
                "no-selected-gate negatives or selected gates with at least one visible "
                "corner that pass distance and projected-size thresholds"
            ),
            "maximum_gate_distance_m": self.maximum_gate_distance_m,
            "minimum_projected_width_px": self.minimum_gate_span_px,
            "minimum_projected_height_px": self.minimum_gate_span_px,
            "minimum_projected_area_px2": self.minimum_gate_area_px2,
            "excluded_samples_are_relabelled_invisible": False,
        }

    @staticmethod
    def onboard_state(frame, frame_dt):
        state = frame["vehicle_state"]
        rotation = rotation_from_rpy(state["attitude_rpy_rad"])
        body_velocity = rotation.T @ np.asarray(state["velocity_mps"], dtype=np.float32)
        angular_velocity = np.asarray(state["angular_velocity_rps"], dtype=np.float32)
        gravity_body = rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        return np.concatenate((
            body_velocity,
            angular_velocity,
            gravity_body,
            np.asarray((frame_dt,), dtype=np.float32),
        )).astype(np.float32)

    def __getitem__(self, index):
        trajectory_index, current = self.samples[index]
        trajectory = self.trajectories[trajectory_index]
        images = []
        for frame_index in (current - 1, current):
            path = trajectory["dir"] / f"rgb_{frame_index:04d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            images.append(image.astype(np.float32) / 255.0)
        targets = trajectory["targets"]
        geometry = trajectory["gate_geometry"]
        sample = {
            "images": np.stack(images),
            "onboard_state": self.onboard_state(
                trajectory["frames"][current], targets["frame_dt_s_f32"][current]
            ),
            "inverse_ttc": targets["inverse_ttc_s_f16"][current].astype(np.float32),
            "ttc_valid": targets["ttc_valid_u8"][current].astype(bool),
            "ttc_approaching": targets["ttc_approaching_u8"][current].astype(bool),
            "inverse_depth": targets["inverse_depth_m_f16"][current].astype(np.float32),
            "depth_valid": targets["depth_valid_u8"][current].astype(bool),
            "flow": targets["backward_rigid_flow_f16"][current].astype(np.float32),
            "flow_valid": targets["flow_valid_u8"][current].astype(bool),
            "gate_corners_px": targets["gate_corners_px_f32"][current].astype(np.float32),
            "gate_corners_visible": targets["gate_corners_valid_u8"][current].astype(bool),
            "gate_index": np.int64(targets["gate_index_i16"][current]),
            "gate_supervision_eligible": np.bool_(geometry["eligible"][current]),
            "gate_distance_m": np.float32(geometry["distance_m"][current]),
            "gate_projected_width_px": np.float32(geometry["width_px"][current]),
            "gate_projected_height_px": np.float32(geometry["height_px"][current]),
            "gate_projected_area_px2": np.float32(geometry["area_px2"][current]),
        }
        if self.augment:
            self._augment(sample)
        result = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else torch.tensor(value)
            for key, value in sample.items()
        }
        result.update(
            layout_id=trajectory["layout_id"],
            scenario=trajectory["scenario"],
            trajectory_id=trajectory["trajectory_id"],
            trajectory_type=trajectory["trajectory_type"],
            frame_index=int(current),
        )
        return result

    @staticmethod
    def _augment(sample):
        gain = np.random.uniform(0.75, 1.25)
        gamma = np.random.uniform(0.8, 1.2)
        sample["images"] = np.clip(gain * sample["images"], 0.0, 1.0) ** gamma
        if np.random.random() >= 0.5:
            return
        for key in (
            "images", "inverse_ttc", "ttc_valid", "ttc_approaching",
            "inverse_depth", "depth_valid", "flow", "flow_valid",
        ):
            sample[key] = np.ascontiguousarray(sample[key][..., ::-1])
        sample["flow"][0] *= -1.0
        sample["onboard_state"][[1, 3, 5, 7]] *= -1.0
        sample["gate_corners_px"][:, 0] = 159.0 - sample["gate_corners_px"][:, 0]
        reorder = [1, 0, 3, 2]
        sample["gate_corners_px"] = np.ascontiguousarray(sample["gate_corners_px"][reorder])
        sample["gate_corners_visible"] = np.ascontiguousarray(
            sample["gate_corners_visible"][reorder]
        )


def gate_sampling_weights(dataset: TTCGateDataset):
    """Balance eligible strata, then trajectory type, trajectory, and frame."""
    target_mass = np.asarray((0.20, 0.10, 0.30, 0.10, 0.30), dtype=np.float64)
    counts, eligible, types, trajectory_keys = [], [], [], []
    for trajectory_index, current in dataset.samples:
        trajectory = dataset.trajectories[trajectory_index]
        counts.append(int(trajectory["targets"]["gate_corners_valid_u8"][current].sum()))
        eligible.append(bool(trajectory["gate_geometry"]["eligible"][current]))
        types.append(trajectory["trajectory_type"])
        trajectory_keys.append(f"{trajectory['layout_id']}/{trajectory['trajectory_id']}")
    counts = np.asarray(counts, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    types = np.asarray(types)
    trajectory_keys = np.asarray(trajectory_keys)
    probability = np.zeros(len(dataset), dtype=np.float64)
    for visible_count, mass in enumerate(target_mass):
        stratum = eligible & (counts == visible_count)
        eligible_types = np.unique(types[stratum])
        if not len(eligible_types):
            continue
        for trajectory_type in eligible_types:
            typed = stratum & (types == trajectory_type)
            keys = np.unique(trajectory_keys[typed])
            for key in keys:
                selected = typed & (trajectory_keys == key)
                probability[selected] = (
                    mass / len(eligible_types) / len(keys) / int(selected.sum())
                )
    probability /= probability.sum()
    summary = {
        "natural_visible_count": {
            str(index): int((counts == index).sum()) for index in range(5)
        },
        "eligible_visible_count": {
            str(index): int((eligible & (counts == index)).sum()) for index in range(5)
        },
        "eligible_samples": int(eligible.sum()),
        "excluded_samples": int((~eligible).sum()),
        "gate_supervision_policy": dataset.gate_supervision_policy(),
        "target_visible_count_mass": {
            str(index): float(value) for index, value in enumerate(target_mass)
        },
        "expected_visible_count_mass": {
            str(index): float(probability[counts == index].sum()) for index in range(5)
        },
    }
    return torch.from_numpy(probability.astype(np.float64)), summary
