#!/usr/bin/env python3
"""Convert a modern PyTorch/QAT checkpoint into a version-neutral NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.quantization import prepare_int8_qat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gate_head = state.get("gate_head", True)
    base = Gap8MultiTaskNet(gate_head=gate_head)
    if state.get("quantization_aware"):
        qat = prepare_int8_qat(Gap8MultiTaskNet(gate_head=gate_head))
        qat.load_state_dict(state["model"])
        qat.eval()
        with torch.no_grad():
            for module in qat.modules():
                if isinstance(module, nn.qat.Conv2d):
                    module.weight.copy_(module.weight_fake_quant(module.weight))
        compatible = {
            key: value
            for key, value in qat.state_dict().items()
            if key in base.state_dict()
        }
        base.load_state_dict(compatible, strict=False)
    else:
        base.load_state_dict(state["model"])
    tensors = {
        key: value.detach().cpu().numpy()
        for key, value in base.state_dict().items()
    }
    np.savez_compressed(args.output / "packed_float_state.npz", **tensors)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "source_quantization_aware": bool(state.get("quantization_aware")),
        "architecture": "Gap8PackedMultiTaskNet-rf123-nemoqat-v6",
        "gate_head": gate_head,
        "packed_channels": base.packed_channels,
        "state_keys": sorted(tensors),
        "normalization": "CNN expects float HM01B0/255; NEMO eps_in=1/255",
    }
    (args.output / "bridge_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
