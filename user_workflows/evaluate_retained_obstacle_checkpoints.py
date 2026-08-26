#!/usr/bin/env python3
"""Compare retained ESPNet checkpoints on untouched obstacle test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ISAACSIM_REPO = Path("/home/cchen/isaacsim-workspace")
sys.path.insert(0, str(ISAACSIM_REPO))

from gap8_perception.temporal_data import TemporalHorizonDataset  # noqa: E402
from train_retained_obstacle_gate import (  # noqa: E402
    RetainedObstacleGateModel,
    evaluate_obstacle,
)


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "obstacle-dataset", "obstacle-checkpoint", "gate-checkpoint", "output"
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = json.loads(
        (args.obstacle_dataset / "dataset_manifest.json").read_text()
    )["camera_calibration"]
    model = RetainedObstacleGateModel(
        args.obstacle_checkpoint, args.gate_checkpoint, camera
    ).to(device)
    datasets = {
        split: TemporalHorizonDataset(
            args.obstacle_dataset, split, 2, minimum_current_index=2
        )
        for split in ("validation", "test")
    }
    loaders = {
        split: DataLoader(
            dataset, 32, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    reports = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        reports.append({
            "checkpoint": str(path),
            "epoch": int(checkpoint["epoch"]),
            "validation": evaluate_obstacle(model, loaders["validation"], device),
            "test": evaluate_obstacle(model, loaders["test"], device),
        })
        print(json.dumps(reports[-1]), flush=True)
    output = {"selection_uses_validation_only": True, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
