#!/usr/bin/env python
"""Python-3.7 NEMO integer export for the two-frame DORY student."""

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

    def forward(self, frames):
        return self.stage1(self.stem(frames))


class FineHeadNet(nn.Module):
    input_shape = (32, 40, 40)

    def __init__(self, channels):
        super(FineHeadNet, self).__init__()
        self.output_channels = channels
        self.head_features = DSConv(32, 16)
        self.output_proj = ConvBNReLU(16, channels, 1)

    def forward_features(self, shared):
        return self.head_features(shared)

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


class DangerHeadNet(nn.Module):
    input_shape = (32, 40, 40)
    output_channels = 1

    def __init__(self):
        super(DangerHeadNet, self).__init__()
        self.stage2 = Stage(32, 64, 3)
        self.stage3 = Stage(64, 96, 7)
        self.output_proj = ConvBNReLU(96, 1, 1)

    def forward_features(self, shared):
        return self.stage3(self.stage2(shared))

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


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
            if image is None or image.shape != (160, 160):
                raise RuntimeError("invalid HM01B0 frame %s" % path)
            frames.append(image)
        values.append(np.stack(frames, axis=0))
    return torch.from_numpy(np.stack(values)).float() / 255.0


def sampled(values, count):
    indices = np.linspace(0, len(values) - 1, min(count, len(values))).astype(int)
    return [values[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--parity-images", type=int, default=512)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    parser.add_argument("--qat-directory", type=Path)
    args = parser.parse_args()
    base.image_tensor = paired_image_tensor
    calibration = sampled(temporal_pairs(args.dataset, "train"), args.calibration_images)
    parity = sampled(temporal_pairs(args.dataset, "test"), args.parity_images)

    encoder = EncoderNet()
    base.load_archive(encoder, args.bridge / "encoder_float_state.npz")
    encoder_integer, encoder_float, encoder_report = base.quantize_encoder(
        encoder, calibration, parity, args.output / "encoder",
        args.residual_requantization_factor,
        qat_checkpoint=(
            args.qat_directory / "encoder_qat.pt" if args.qat_directory else None
        ),
    )
    reports = [encoder_report]
    for name, model in (
        ("corner_head", FineHeadNet(4)),
        ("gate_head", FineHeadNet(1)),
        ("danger_head", DangerHeadNet()),
    ):
        base.load_head(model, args.bridge / (name + "_float_state.npz"), name)
        reports.append(base.quantize_head(
            name, model, encoder_integer, encoder_float,
            encoder_report["output_epsilon"], calibration, parity,
            args.output / name, args.residual_requantization_factor,
            qat_checkpoint=(
                args.qat_directory / (name + "_qat.pt")
                if args.qat_directory else None
            ),
        ))
    report = {
        "format": "espnet-dory-student-nemo-v1",
        "architecture": "ESPNetDoryStudent",
        "temporal_input_order": ["previous", "current"],
        "input_layout": "HWC with two interleaved uint8 channels in DORY",
        "graphs": reports,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nemo_espnet_dory_student_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
