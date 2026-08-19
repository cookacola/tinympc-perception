#!/usr/bin/env python3
"""Evaluate one mixed checkpoint on a specified gate/no-gate test domain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/home/cchen/isaacsim-workspace")

import torch
from torch.utils.data import DataLoader

from gap8_perception.data import MultiTaskDataset
from gap8_perception.temporal_data import TemporalHorizonDataset
from gap8_perception.train_encoder_ablation import evaluate as evaluate_obstacle
from train_retained_obstacle_gate import (
    NoGateDataset,
    RealGateDataset,
    RetainedObstacleGateModel,
    evaluate_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "obstacle-dataset", "gate-dataset", "gate-targets", "gate-split-file",
        "no-gate-dataset", "real-root", "obstacle-checkpoint", "gate-checkpoint",
        "mixed-checkpoint", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = json.loads(
        (arguments.obstacle_dataset / "dataset_manifest.json").read_text()
    )["camera_calibration"]
    model = RetainedObstacleGateModel(
        arguments.obstacle_checkpoint, arguments.gate_checkpoint, camera
    ).to(device)
    checkpoint = torch.load(
        arguments.mixed_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    def loader(dataset, batch_size=arguments.batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=arguments.workers,
            pin_memory=True,
            persistent_workers=arguments.workers > 0,
        )

    obstacle = loader(
        TemporalHorizonDataset(
            arguments.obstacle_dataset,
            "test",
            2,
            augment=False,
            minimum_current_index=2,
        ),
        32,
    )
    synthetic = loader(MultiTaskDataset(
        arguments.gate_dataset,
        arguments.gate_targets,
        arguments.gate_split_file,
        "test",
    ))
    negative = loader(NoGateDataset(arguments.no_gate_dataset, (4,)))
    real = loader(RealGateDataset(arguments.real_root, ("flight_08",)))
    result = {
        "checkpoint": str(arguments.mixed_checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "gate_dataset": str(arguments.gate_dataset),
        "no_gate_dataset": str(arguments.no_gate_dataset),
        "obstacle_test": evaluate_obstacle(model, obstacle, device),
        "gate_test": evaluate_gate(model, synthetic, negative, real, device),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
