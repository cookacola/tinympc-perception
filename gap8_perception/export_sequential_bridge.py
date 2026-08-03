#!/usr/bin/env python3
"""Bridge a sequential float/QAT checkpoint into a NeMO-compatible NPZ state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model_sequential import SequentialSTDCNet
from .quantization import prepare_int8_qat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") != "SequentialSTDCNet":
        raise RuntimeError("checkpoint is not SequentialSTDCNet")
    base = SequentialSTDCNet()
    qat = bool(state.get("quantization_aware"))
    if qat:
        quantized = prepare_int8_qat(base)
        quantized.load_state_dict(state["model"])
        # Transfer fake-quantized weights and the non-quantized BN state to
        # the simple model whose state-key ABI is shared with Python 3.7.
        source = quantized.state_dict()
        target = base.state_dict()
        for name, module in quantized.named_modules():
            if isinstance(module, nn.qat.Conv2d):
                target[f"{name}.weight"] = module.weight_fake_quant(module.weight).detach().cpu()
        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                target[key] = value.detach().cpu()
        base.load_state_dict(target)
    else:
        base.load_state_dict(state["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().numpy() for key, value in base.state_dict().items()}
    np.savez_compressed(args.output / "sequential_float_state.npz", **tensors)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "source_qat": qat,
        "architecture": "SequentialSTDCNet",
        "input": [1, 120, 160],
        "output": [12, 15, 20],
        "state_keys": sorted(tensors),
    }
    (args.output / "bridge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
