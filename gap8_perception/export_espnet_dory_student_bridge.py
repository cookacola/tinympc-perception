#!/usr/bin/env python3
"""Export a trained ESPNet DORY student as version-neutral graph archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gap8_perception.model_espnet_dory_student import (
    build_student,
    deployment_graphs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    model = build_student(architecture).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    shapes = {
        "encoder": ([2, 160, 160], [32, 40, 40]),
        "corner_head": ([32, 40, 40], [4, 40, 40]),
        "gate_head": ([32, 40, 40], [1, 40, 40]),
        "danger_head": ([32, 40, 40], [1, 10, 10]),
    }
    report = {
        "format": "espnet-dory-student-bridge-v1",
        "source_checkpoint": str(args.checkpoint),
        "source_epoch": int(checkpoint["epoch"]),
        "architecture": checkpoint["architecture"],
        "temporal_input": "two consecutive HM01B0 frames, oldest then current",
        "graphs": {},
    }
    for name, graph in deployment_graphs(model).items():
        tensors = {
            key: value.detach().cpu().numpy()
            for key, value in graph.state_dict().items()
        }
        np.savez_compressed(args.output / f"{name}_float_state.npz", **tensors)
        report["graphs"][name] = {
            "input": shapes[name][0],
            "output": shapes[name][1],
            "state_keys": sorted(tensors),
        }
    (args.output / "bridge_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
