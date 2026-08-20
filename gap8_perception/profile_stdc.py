#!/usr/bin/env python3
"""Static parameter/MAC/activation report for Gap8STDCMultiHeadNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model_stdc import Gap8STDCMultiHeadNet


def profile(model: nn.Module) -> dict:
    macs = 0
    peak_elements = 0
    layers = []
    handles = []

    def hook(name):
        def record(module, inputs, output):
            nonlocal macs, peak_elements
            if not isinstance(output, torch.Tensor):
                return
            elements = output.numel()
            peak_elements = max(peak_elements, elements)
            if isinstance(module, nn.Conv2d):
                batch, cout, height, width = output.shape
                cin_per_group = module.in_channels // module.groups
                layer_macs = (
                    batch
                    * cout
                    * height
                    * width
                    * cin_per_group
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
                macs += layer_macs
                layers.append({"name": name, "macs": layer_macs, "shape": list(output.shape)})
        return record

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(torch.zeros(1, *model.input_shape, device=device))
    for handle in handles:
        handle.remove()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    output_shapes = (
        {key: list(value.shape) for key, value in outputs.items()}
        if isinstance(outputs, dict)
        else {"output": list(outputs.shape)}
    )
    return {
        "architecture": type(model).__name__,
        "input_nchw": [1, *model.input_shape],
        "outputs": output_shapes,
        "parameters": parameters,
        "macs": macs,
        # This is a single-tensor lower bound, not a DORY liveness/tiling claim.
        "largest_single_activation_int8_bytes": peak_elements,
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = profile(Gap8STDCMultiHeadNet().eval())
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
