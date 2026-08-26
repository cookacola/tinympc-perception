#!/usr/bin/env python3
"""Export the selected ESPNet/DroNet/gate checkpoint as fixed graph archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model_espnet_dronet_gate import ESPNetDroNetGate, deployment_graphs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "espnet-dronet-middle-gate-v1":
        raise ValueError("unexpected checkpoint format")
    model = ESPNetDroNetGate().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    shapes = {
        "encoder": ([2, 160, 160], [64, 20, 20]),
        "corner_head": ([64, 20, 20], [4, 20, 20]),
        "gate_head": ([64, 20, 20], [1, 20, 20]),
        "presence_head": ([64, 20, 20], [1]),
        "navigation_head": ([64, 20, 20], [2]),
    }
    summary = json.loads(args.summary.read_text())
    report = {
        "format": "espnet-dronet-middle-gate-bridge-v1",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_epoch": int(checkpoint["epoch"]),
        "temporal_input": "two consecutive HM01B0 frames, previous then current",
        "gate_tap": "middle_stage2",
        "gate_confidence_fusion": summary["structured_confidence_fusion"],
        "navigation_collision_threshold": summary["navigation_collision_threshold"],
        "graphs": {},
    }
    for name, graph in deployment_graphs(model).items():
        tensors = {key: value.detach().cpu().numpy() for key, value in graph.state_dict().items()}
        np.savez_compressed(args.output / f"{name}_float_state.npz", **tensors)
        report["graphs"][name] = {
            "input": shapes[name][0], "output": shapes[name][1],
            "state_keys": sorted(tensors),
        }
    (args.output / "bridge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
