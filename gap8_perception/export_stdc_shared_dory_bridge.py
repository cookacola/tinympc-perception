#!/usr/bin/env python3
"""Bridge a shared-DORY checkpoint into Python-3.7-compatible NPZ graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gap8_perception.model_stdc_dory import (
    Gap8STDCSharedDoryNet,
    shared_deployment_graphs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") != "Gap8STDCSharedDoryNet":
        raise RuntimeError("checkpoint is not a Gap8STDCSharedDoryNet")
    model = Gap8STDCSharedDoryNet()
    model.load_state_dict(state["model"])
    names = ("encoder", "corner_head", "danger_head")
    graphs = shared_deployment_graphs(model)
    shapes = {
        "encoder": ([1, 120, 160], [32, 30, 40]),
        "corner_head": ([32, 30, 40], [4, 30, 40]),
        "danger_head": ([32, 30, 40], [1, 8, 10]),
    }
    report = {
        "source_checkpoint": str(args.checkpoint),
        "architecture": state["architecture"],
        "graphs": {},
    }
    for name, graph in zip(names, graphs):
        tensors = {
            key: value.detach().cpu().numpy()
            for key, value in graph.state_dict().items()
        }
        np.savez_compressed(args.output / f"{name}_float_state.npz", **tensors)
        input_shape, output_shape = shapes[name]
        report["graphs"][name] = {
            "input": input_shape,
            "output": output_shape,
            "state_keys": sorted(tensors),
        }
    (args.output / "bridge_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
