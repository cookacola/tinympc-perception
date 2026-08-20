#!/usr/bin/env python3
"""Convert a modern DORY-pair checkpoint into Python-3.7-compatible NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gap8_perception.model_stdc_dory import (
    Gap8STDCCornerDoryNet,
    Gap8STDCDangerDoryNet,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") != "Gap8STDCDoryPair":
        raise RuntimeError("checkpoint is not a Gap8STDCDoryPair")
    report = {
        "source_checkpoint": str(args.checkpoint),
        "architecture": state["architecture"],
        "graphs": {},
    }
    for name, model, key in (
        ("corner", Gap8STDCCornerDoryNet(), "corner_model"),
        ("danger", Gap8STDCDangerDoryNet(), "danger_model"),
    ):
        model.load_state_dict(state[key])
        tensors = {
            tensor_name: value.detach().cpu().numpy()
            for tensor_name, value in model.state_dict().items()
        }
        np.savez_compressed(args.output / f"{name}_float_state.npz", **tensors)
        report["graphs"][name] = {
            "input": [1, 120, 160],
            "output": [4, 30, 40] if name == "corner" else [1, 8, 10],
            "state_keys": sorted(tensors),
        }
    (args.output / "bridge_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
