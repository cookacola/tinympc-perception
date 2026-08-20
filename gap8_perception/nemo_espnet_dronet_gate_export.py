#!/usr/bin/env python
"""Python-3.7 NEMO export for middle-tap ESPNet gate and DroNet heads."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from gap8_perception import nemo_stdc_shared_dory_export as base
from gap8_perception.nemo_stdc_dory_export import ConvBNReLU, DSConv, Stage


class EncoderNet(nn.Module):
    input_shape = (2, 160, 160)

    def __init__(self):
        super(EncoderNet, self).__init__()
        self.stem = nn.Sequential(ConvBNReLU(2, 16, 3, 2), DSConv(16, 16))
        self.stage1 = Stage(16, 32, 2)
        self.stage2 = Stage(32, 64, 3)

    def forward(self, frames):
        return self.stage2(self.stage1(self.stem(frames)))


def terminal(cin, cout, kernel):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=0, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=False),
    )


class MiddleSpatialHead(nn.Module):
    input_shape = (64, 20, 20)

    def __init__(self, channels):
        super(MiddleSpatialHead, self).__init__()
        self.output_channels = channels
        self.adapter = ConvBNReLU(64, 32, 1)
        self.head_features = DSConv(32, 16)
        self.output_proj = terminal(16, channels, 1)

    def forward_features(self, shared):
        return self.head_features(self.adapter(shared))

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


class PresenceHead(nn.Module):
    input_shape = (64, 20, 20)
    output_channels = 1

    def __init__(self):
        super(PresenceHead, self).__init__()
        self.pool = nn.AvgPool2d(20)
        self.output_proj = terminal(64, 1, 1)

    def forward_features(self, shared):
        return shared

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.pool(self.output_proj[1](self.output_proj[0](features)))

    def forward(self, shared):
        # The shifted terminal activation is nonnegative, so averaging and the
        # affine 1x1 projection commute. Project at 20x20 and pool afterward to
        # avoid the GAP8 scalar-convolution kernel edge case.
        return self.pool(self.output_proj(self.forward_features(shared)))


class NavigationHead(nn.Module):
    input_shape = (64, 20, 20)
    output_channels = 2

    def __init__(self):
        super(NavigationHead, self).__init__()
        self.stage3 = Stage(64, 96, 7)
        self.pool = nn.AvgPool2d(10)
        self.output_proj = terminal(96, 2, 1)

    def forward_features(self, shared):
        return self.stage3(shared)

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.pool(self.output_proj[1](self.output_proj[0](features)))

    def forward(self, shared):
        return self.pool(self.output_proj(self.forward_features(shared)))


def temporal_pairs(dataset, split):
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    pairs = []
    for layout in manifest["layouts"]:
        if layout["split"] != split:
            continue
        layout_dir = dataset / layout.get("layout_path", layout["layout_id"])
        for trajectory in layout["trajectories"]:
            directory = layout_dir / "trajectories" / trajectory["trajectory_id"]
            images = sorted(directory.glob("rgb_*.png"))
            pairs.extend(zip(images[:-1], images[1:]))
    if not pairs:
        raise RuntimeError("no temporal pairs for split %s" % split)
    return pairs


def paired_image_tensor(pairs):
    values = []
    for previous, current in pairs:
        frames = []
        for path in (previous, current):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError("invalid HM01B0 frame %s" % path)
            if image.shape != (160, 160):
                height, width = image.shape
                if height < 200 or width < 200:
                    raise RuntimeError("invalid HM01B0 frame %s" % path)
                image = image[(height - 200) // 2:(height + 200) // 2,
                              (width - 200) // 2:(width + 200) // 2]
                image = cv2.resize(
                    image, (160, 160), interpolation=cv2.INTER_LINEAR
                )
            frames.append(image)
        values.append(np.stack(frames, axis=0))
    return torch.from_numpy(np.stack(values)).float() / 255.0


def sampled(values, count):
    indices = np.linspace(0, len(values) - 1, min(count, len(values))).astype(int)
    return [values[index] for index in indices]


def identity_terminal(state, prefix, bias):
    eps = 1.0e-5
    state[prefix + ".1.weight"].fill_((1.0 + eps) ** 0.5)
    state[prefix + ".1.bias"].copy_(bias)
    state[prefix + ".1.running_mean"].zero_()
    state[prefix + ".1.running_var"].fill_(1.0)
    state[prefix + ".1.num_batches_tracked"].zero_()


def load_spatial_head(model, archive_path):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    for key in archive.files:
        if key.startswith("adapter."):
            state[key] = torch.from_numpy(archive[key])
        elif key.startswith("head.0."):
            state[key.replace("head.0.", "head_features.", 1)] = torch.from_numpy(archive[key])
    state["output_proj.0.weight"] = torch.from_numpy(archive["head.1.weight"])
    identity_terminal(state, "output_proj", torch.from_numpy(archive["head.1.bias"]))
    model.load_state_dict(state, strict=True)


def load_gate_head(model, archive_path):
    """Equalize wide mask logits before uint8 export; firmware restores x32."""
    load_spatial_head(model, archive_path)
    with torch.no_grad():
        model.output_proj[0].weight.div_(32.0)
        model.output_proj[1].bias.div_(32.0)


def load_presence_head(model, archive_path):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    linear = torch.from_numpy(archive["head.weight"])
    # Shrink the deployment logit range before unsigned int8 quantization.
    # Firmware multiplies the decoded logit by the same exact constant.
    state["output_proj.0.weight"] = linear[:, :, None, None] / 4.0
    identity_terminal(state, "output_proj", torch.from_numpy(archive["head.bias"]) / 4.0)
    model.load_state_dict(state, strict=True)


def load_navigation_head(model, archive_path):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    for key in archive.files:
        if key.startswith("stage3."):
            state[key] = torch.from_numpy(archive[key])
    linear = torch.from_numpy(archive["head.weight"])
    # Yaw and collision share a terminal tensor but collision logits span a
    # much wider range. Equalize only the collision channel for int8, then
    # restore it after decoding. This changes neither the float function nor
    # the number of deployed MACs.
    linear[1].div_(8.0)
    state["output_proj.0.weight"] = linear[:, :, None, None]
    identity_terminal(state, "output_proj", torch.zeros(2))
    model.load_state_dict(state, strict=True)


def qat_checkpoint(directory, graph):
    if directory is None:
        return None
    path = directory / (graph + "_qat.pt")
    return path if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--parity-images", type=int, default=512)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    parser.add_argument("--qat-directory", type=Path)
    parser.add_argument("--bn-calibration-range-factor", type=int, default=8)
    parser.add_argument("--calibrate-batchnorm", action="store_true")
    parser.add_argument("--canonicalize-batchnorm", action="store_true")
    parser.add_argument("--signed-weight-bits", type=int, choices=(7, 8), default=7)
    args = parser.parse_args()
    bridge_report = json.loads((args.bridge / "bridge_report.json").read_text())
    if bridge_report["format"] != "espnet-dronet-middle-gate-bridge-v1":
        raise RuntimeError("unexpected bridge format")
    base.image_tensor = paired_image_tensor
    calibration = sampled(temporal_pairs(args.dataset, "train"), args.calibration_images)
    parity = sampled(temporal_pairs(args.dataset, "test"), args.parity_images)

    encoder = EncoderNet()
    base.load_archive(encoder, args.bridge / "encoder_float_state.npz")
    if args.canonicalize_batchnorm:
        base.canonicalize_batchnorm_affine(encoder)
    encoder_integer, encoder_float, encoder_report = base.quantize_encoder(
        encoder, calibration, parity, args.output / "encoder",
        args.residual_requantization_factor,
        qat_checkpoint=qat_checkpoint(args.qat_directory, "encoder"),
        bn_calibration_range_factor=args.bn_calibration_range_factor,
        calibrate_batchnorm=args.calibrate_batchnorm,
        signed_weight_bits=args.signed_weight_bits,
    )
    reports = [encoder_report]
    heads = (
        ("corner_head", MiddleSpatialHead(4), load_spatial_head),
        ("gate_head", MiddleSpatialHead(1), load_gate_head),
        ("presence_head", PresenceHead(), load_presence_head),
        ("navigation_head", NavigationHead(), load_navigation_head),
    )
    for name, model, loader in heads:
        loader(model, args.bridge / (name + "_float_state.npz"))
        if args.canonicalize_batchnorm:
            base.canonicalize_batchnorm_affine(model)
        reports.append(base.quantize_head(
            name, model, encoder_integer, encoder_float,
            encoder_report["output_epsilon"], calibration, parity,
            args.output / name, args.residual_requantization_factor,
            qat_checkpoint=qat_checkpoint(args.qat_directory, name),
            bn_calibration_range_factor=args.bn_calibration_range_factor,
            calibrate_batchnorm=args.calibrate_batchnorm,
            signed_weight_bits=args.signed_weight_bits,
        ))
    report = {
        "format": "espnet-dronet-middle-gate-nemo-v1",
        "temporal_input_order": ["previous", "current"],
        "input_layout": "HWC with two interleaved uint8 channels in DORY",
        "bn_calibration_range_factor": args.bn_calibration_range_factor,
        "data_driven_batchnorm_calibration": args.calibrate_batchnorm,
        "canonical_batchnorm_affine": args.canonicalize_batchnorm,
        "signed_weight_bits": args.signed_weight_bits,
        "graphs": reports,
        "deployment_logit_scale": {
            "corner_head": [1.0, 1.0, 1.0, 1.0],
            "gate_head": [32.0],
            "presence_head": [4.0],
            "navigation_head": [1.0, 8.0],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nemo_espnet_dronet_gate_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
