#!/usr/bin/env python3
"""Bridge a trained DORY gate/TTC checkpoint to Python-3.7 NeMO archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_dory_model import DoryPartitionedMotionGateTTCNet


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state(module, path):
    np.savez_compressed(
        path,
        **{key: value.detach().cpu().numpy() for key, value in module.state_dict().items()},
    )


def sampled_records(dataset, count):
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset))).astype(int)
    records = []
    for dataset_index in indices:
        trajectory_index, current = dataset.samples[dataset_index]
        trajectory = dataset.trajectories[trajectory_index]
        frame = trajectory["frames"][current]
        dt = trajectory["targets"]["frame_dt_s_f32"][current]
        state = dataset.onboard_state(frame, dt)
        records.append({
            "previous": str((trajectory["dir"] / f"rgb_{current - 1:04d}.png").resolve()),
            "current": str((trajectory["dir"] / f"rgb_{current:04d}.png").resolve()),
            "normalized_state": np.clip(
                state / np.asarray(DoryPartitionedMotionGateTTCNet.onboard_scale), -4.0, 4.0
            ).tolist(),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--parity-samples", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model", saved)
    blocks = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("ttc_head.deep.") and ".block." in key
    }
    refinements = max(blocks) + 1
    model = DoryPartitionedMotionGateTTCNet(ttc_refinements=refinements).eval()
    model.load_state_dict(state)
    save_state(model.encoder, args.output / "encoder_float_state.npz")
    save_state(model.gate_head, args.output / "gate_head_float_state.npz")
    save_state(model.ttc_head, args.output / "ttc_head_float_state.npz")
    records = {
        "calibration": sampled_records(
            TTCGateDataset(args.dataset, "train"), args.calibration_samples
        ),
        "parity": sampled_records(
            TTCGateDataset(args.dataset, "test"), args.parity_samples
        ),
    }
    (args.output / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
    report = {
        "format": "dory-motion-gate-ttc-bridge-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_epoch": saved.get("epoch"),
        "ttc_refinements": refinements,
        "state_shift": 4.0,
        "graphs": {
            "encoder": {"input": [2, 160, 160], "output": [64, 20, 20]},
            "gate_head": {"input": [64, 20, 20], "output": [8, 20, 20]},
            "ttc_head": {"input": [74, 20, 20], "output": [7, 20, 20]},
        },
        "sample_counts": {key: len(value) for key, value in records.items()},
    }
    (args.output / "bridge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
