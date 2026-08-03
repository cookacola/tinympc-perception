"""Static resource report for the replacement sequential student."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .model_sequential import SequentialSTDCNet


def profile(model: nn.Module | None = None) -> dict:
    model = (model or SequentialSTDCNet()).eval()
    macs = 0
    layers = []
    handles = []

    def hook(name):
        def record(module, inputs, output):
            nonlocal macs
            if not isinstance(output, torch.Tensor):
                return
            _, cout, height, width = output.shape
            cin = module.in_channels // module.groups
            count = cout * height * width * cin * module.kernel_size[0] * module.kernel_size[1]
            macs += count
            layers.append({"name": name, "macs": count, "output": [cout, height, width]})
        return record

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        output = model(torch.zeros(1, *model.input_shape))
    for handle in handles:
        handle.remove()
    return {
        "architecture": type(model).__name__,
        "input_nchw": [1, *model.input_shape],
        "output_nchw": list(output.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs": macs,
        "largest_single_activation_int8_bytes": max(
            [item["output"][0] * item["output"][1] * item["output"][2] for item in layers]
            + [0]
        ),
        "layers": layers,
        "operators": ["Conv2d", "BatchNorm2d(training-only)", "ReLU", "Conv2d(linear terminal)"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = profile()
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
