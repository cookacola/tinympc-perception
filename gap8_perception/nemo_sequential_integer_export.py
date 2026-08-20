#!/usr/bin/env python
"""Pinned Python-3.7 NeMO export for the sequential 12-channel student."""
from __future__ import print_function

import argparse
import copy
import json
from pathlib import Path

import cv2
import nemo
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel, stride=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
        )


class DSBlock(nn.Sequential):
    def __init__(self, channels):
        super(DSBlock, self).__init__(
            ConvBNReLU(channels, channels, 3, groups=channels),
            ConvBNReLU(channels, channels, 1),
        )


class SequentialNet(nn.Module):
    def __init__(self):
        super(SequentialNet, self).__init__()
        self.stem = ConvBNReLU(1, 16, 3, stride=2)
        self.layers = nn.Sequential(
            ConvBNReLU(16, 16, 3, groups=16), ConvBNReLU(16, 24, 1),
            ConvBNReLU(24, 24, 3, stride=2, groups=24), ConvBNReLU(24, 32, 1),
            ConvBNReLU(32, 32, 3, groups=32), ConvBNReLU(32, 48, 1),
            ConvBNReLU(48, 48, 3, stride=2, groups=48), ConvBNReLU(48, 64, 1),
            ConvBNReLU(64, 64, 3, groups=64), ConvBNReLU(64, 96, 1),
            *[DSBlock(96) for _ in range(6)]
        )
        # DORY stores terminal activations as uint8.  The affine shift maps
        # the logical score interval [-6, 6] into the nonnegative PACT domain;
        # firmware subtracts this offset after applying output epsilon.
        self.output_proj = ConvBNReLU(96, 12, 1)

    def forward(self, image):
        return self.output_proj(self.layers(self.stem(image)))


def load_archive(model, path):
    archive = np.load(str(path))
    state = model.state_dict()
    for key in archive.files:
        if key == "output.weight":
            continue
        if key in state:
            state[key] = torch.from_numpy(archive[key])
    state["output_proj.0.weight"] = torch.from_numpy(archive["output.weight"])
    eps = model.output_proj[1].eps
    state["output_proj.1.weight"] = torch.full_like(state["output_proj.1.weight"], (1.0 + eps) ** 0.5)
    state["output_proj.1.bias"] = torch.full_like(state["output_proj.1.bias"], 6.0)
    state["output_proj.1.running_mean"].zero_()
    state["output_proj.1.running_var"].fill_(1.0)
    model.load_state_dict(state)


def image_tensor(paths):
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[20:140] for path in paths]
    return torch.from_numpy(np.asarray(images)).unsqueeze(1).float() / 255.0


def int8_weights(model):
    report = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                values = torch.round(module.weight).clamp(-128, 127)
                module.weight.copy_(values)
                report.append({"name": name, "minimum": int(values.min()), "maximum": int(values.max())})
    return report


def verify_weights(path):
    graph = onnx.load(str(path))
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Conv":
            value = initializers[node.input[1]]
            if not np.array_equal(value, np.rint(value)) or value.min() < -128 or value.max() > 127:
                raise RuntimeError("non-int8 convolution initializer %s" % node.input[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.split_file:
        with args.split_file.open("r") as stream:
            split = json.load(stream)
        train_spec = split["train"]
        if isinstance(train_spec, dict):
            names = [
                "shard_%09d" % index
                for index in range(
                    int(train_spec["start"]),
                    int(train_spec["stop"]),
                    int(train_spec["step"]),
                )
            ]
        else:
            names = train_spec
        paths = []
        for name in names:
            paths.extend(sorted((args.dataset / name).glob("hm01b0_mono_*.png")))
    else:
        paths = sorted(args.dataset.glob("shard_*/hm01b0_mono_*.png"))
    if not paths:
        raise RuntimeError("no calibration images")
    indices = np.linspace(0, len(paths) - 1, min(len(paths), args.calibration_images)).astype(int)
    calibration = [paths[index] for index in indices]
    model = SequentialNet().eval()
    load_archive(model, args.bridge / "sequential_float_state.npz")
    model_float = copy.deepcopy(model).eval()
    model = nemo.transform.quantize_pact(model, dummy_input=torch.ones(1, 1, 120, 160))
    model.change_precision(bits=8)
    model.reset_alpha_weights()
    model.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration), 32):
            model(image_tensor(calibration[start:start + 32]))
    model.unset_statistics_act()
    model.reset_alpha_act()
    model.qd_stage(eps_in=1.0 / 255.0)
    model.id_stage()
    weight_report = int8_weights(model)
    first = cv2.imread(str(calibration[0]), cv2.IMREAD_GRAYSCALE)[20:140]
    integer_input = torch.from_numpy(first).unsqueeze(0).unsqueeze(0).float()
    golden = []
    hooks = []
    for module in model.modules():
        if module.__class__.__name__ in ("PACT_Act", "PACT_IntegerAct"):
            hooks.append(module.register_forward_hook(
                lambda module, inputs, value: golden.append(value.detach())
            ))
    with torch.no_grad():
        integer_output = model(integer_input)
    for hook in hooks:
        hook.remove()
    np.savetxt(str(args.output / "input.txt"), first.flatten(), fmt="%.3f", delimiter=",", newline=",\n")
    for layer, activation in enumerate(golden):
        values = activation[0].permute(1, 2, 0).numpy().flatten()
        np.savetxt(str(args.output / ("out_layer%d.txt" % layer)), values, fmt="%.3f", delimiter=",", newline=",\n")
    np.savetxt(str(args.output / "output.txt"), integer_output[0].permute(1, 2, 0).numpy().flatten(), fmt="%.3f", delimiter=",", newline=",\n")
    onnx_path = args.output / "sequential_int.onnx"
    nemo.utils.export_onnx(str(onnx_path), model, model, (1, 120, 160), perm=None)
    verify_weights(onnx_path)
    graph = onnx.load(str(onnx_path))
    epsilon_modules = [(name, float(module.eps_out)) for name, module in model.named_modules() if hasattr(module, "eps_out")]
    report = {
        "architecture": "SequentialSTDCNet", "onnx": str(onnx_path),
        "operators": sorted(set(node.op_type for node in graph.graph.node)),
        "integer_output_shape": list(integer_output.shape),
        "integer_output_nonzero": int((integer_output != 0).sum()),
        "golden_activation_files": len(golden),
        "output_epsilon": epsilon_modules[-1][1] if epsilon_modules else None,
        "terminal_score_offset": 6.0,
        "terminal_output_interpretation": "logical_score = uint8 * output_epsilon - 6.0",
        "epsilon_modules": epsilon_modules,
        "weight_quantization": weight_report,
        "calibration_split": "train" if args.split_file else "all_shards_legacy",
    }
    (args.output / "quantization_manifest.json").write_text(json.dumps({
        "tensor": "perception_scores_12x15x20", "shape": [1, 12, 15, 20],
        "signed": False, "bits": 8,
        "scale": report["output_epsilon"], "zero_point": 0,
        "logical_score_offset": 6.0,
        "interpretation": report["terminal_output_interpretation"],
    }, indent=2) + "\n")
    (args.output / "nemo_sequential_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
