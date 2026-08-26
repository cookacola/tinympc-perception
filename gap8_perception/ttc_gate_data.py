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

    def __init__(self, root: str | Path, split: str, augment: bool = False):
        self.root = Path(root)
        self.augment = bool(augment)
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
                })
                self.samples.extend((trajectory_index, current) for current in range(1, len(frames)))

    def __len__(self):
        return len(self.samples)

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
    """Balance visible-count strata, then trajectory type, trajectory, and frame."""
    target_mass = np.asarray((0.20, 0.10, 0.30, 0.10, 0.30), dtype=np.float64)
    counts, types, trajectory_keys = [], [], []
    for trajectory_index, current in dataset.samples:
        trajectory = dataset.trajectories[trajectory_index]
        counts.append(int(trajectory["targets"]["gate_corners_valid_u8"][current].sum()))
        types.append(trajectory["trajectory_type"])
        trajectory_keys.append(f"{trajectory['layout_id']}/{trajectory['trajectory_id']}")
    counts = np.asarray(counts, dtype=np.int64)
    types = np.asarray(types)
    trajectory_keys = np.asarray(trajectory_keys)
    probability = np.zeros(len(dataset), dtype=np.float64)
    for visible_count, mass in enumerate(target_mass):
        stratum = counts == visible_count
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
        "target_visible_count_mass": {
            str(index): float(value) for index, value in enumerate(target_mass)
        },
        "expected_visible_count_mass": {
            str(index): float(probability[counts == index].sum()) for index in range(5)
        },
    }
    return torch.from_numpy(probability.astype(np.float64)), summary
