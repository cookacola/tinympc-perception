#!/usr/bin/env python
"""Python-3.7 NeMO export for DORY-partitioned gate and motion-TTC graphs."""
from __future__ import print_function

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from gap8_perception import nemo_stdc_shared_dory_export as base
from gap8_perception.nemo_stdc_dory_export import (
    ConvBNReLU,
    DSConv,
    ResidualDS,
    Stage,
)


SAMPLES = []
CURRENT_RECORDS = []


def paired_image_tensor(indices):
    global CURRENT_RECORDS
    CURRENT_RECORDS = [SAMPLES[int(index)] for index in indices]
    pairs = []
    for record in CURRENT_RECORDS:
        frames = []
        for name in ("previous", "current"):
            image = cv2.imread(record[name], cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (160, 160):
                raise RuntimeError("invalid two-frame sample %s" % record[name])
            frames.append(image)
        pairs.append(np.stack(frames, axis=0))
    return torch.from_numpy(np.stack(pairs)).float() / 255.0


def state_tensor(shift):
    values = np.asarray(
        [record["normalized_state"] for record in CURRENT_RECORDS], np.float32
    )
    return torch.from_numpy(values + shift)[:, :, None, None].expand(-1, -1, 20, 20)


class EncoderNet(nn.Module):
    input_shape = (2, 160, 160)

    def __init__(self):
        super(EncoderNet, self).__init__()
        self.stem = nn.Sequential(ConvBNReLU(2, 16, 3, 2), DSConv(16, 16))
        self.stage1 = Stage(16, 32, 2)
        self.stage2 = Stage(32, 64, 3)

    def forward(self, images):
        return self.stage2(self.stage1(self.stem(images)))


def terminal(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=False),
    )


def identity_terminal(state, prefix, bias):
    eps = 1.0e-5
    state[prefix + ".1.weight"].fill_((1.0 + eps) ** 0.5)
    state[prefix + ".1.bias"].copy_(bias)
    state[prefix + ".1.running_mean"].zero_()
    state[prefix + ".1.running_var"].fill_(1.0)
    state[prefix + ".1.num_batches_tracked"].zero_()


class GateHead(nn.Module):
    input_shape = (64, 20, 20)
    output_channels = 8

    def __init__(self):
        super(GateHead, self).__init__()
        self.adapter = ConvBNReLU(64, 32, 1)
        self.head_features = DSConv(32, 16)
        self.output_proj = terminal(16, 8)

    def forward_features(self, shared):
        return self.head_features(self.adapter(shared))

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


class TTCHead(nn.Module):
    input_shape = (74, 20, 20)
    output_channels = 7

    def __init__(self, refinements):
        super(TTCHead, self).__init__()
        self.adapter = ConvBNReLU(74, 64, 1)
        self.deep = nn.Sequential(
            *([ResidualDS(64) for _ in range(refinements)] + [DSConv(64, 32)])
        )
        self.shortcut = ConvBNReLU(74, 32, 1)
        self.add = base.nemo.quant.pact.PACT_IntegerAdd()
        self.relu = nn.ReLU(inplace=False)
        self.output_proj = terminal(32, 7)

    def forward_features(self, packed):
        return self.relu(self.add(self.deep(self.adapter(packed)), self.shortcut(packed)))

    def forward_logits(self, packed):
        features = self.forward_features(packed)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, packed):
        return self.output_proj(self.forward_features(packed))


def load_gate(model, archive_path):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    for key in archive.files:
        destination = key
        if key.startswith("output."):
            continue
        if destination in state:
            state[destination] = torch.from_numpy(archive[key])
    state["output_proj.0.weight"] = torch.from_numpy(archive["output.weight"])
    identity_terminal(state, "output_proj", torch.from_numpy(archive["output.bias"]))
    model.load_state_dict(state)


def load_ttc(model, archive_path, state_shift):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    for key in archive.files:
        if key.startswith("output."):
            continue
        if key in state:
            state[key] = torch.from_numpy(archive[key])
    state["output_proj.0.weight"] = torch.from_numpy(archive["output.weight"])
    identity_terminal(state, "output_proj", torch.from_numpy(archive["output.bias"]))
    # The host encodes signed state s as s+shift. Absorb W*shift into the two
    # first-layer BN means so the float function remains exactly unchanged.
    for prefix in ("adapter", "shortcut"):
        weight = state[prefix + ".0.weight"][:, 64:, 0, 0]
        contribution = weight.sum(1) * state_shift
        state[prefix + ".1.running_mean"] += contribution
    model.load_state_dict(state)


class PackedFloatEncoder(nn.Module):
    def __init__(self, encoder, shift):
        super(PackedFloatEncoder, self).__init__()
        self.encoder = encoder
        self.shift = shift

    def forward(self, images):
        e2 = self.encoder(images)
        return torch.cat((e2, state_tensor(self.shift).to(e2)), dim=1)


class PackedIntegerEncoder(nn.Module):
    def __init__(self, encoder, encoder_epsilon, packed_epsilon, shift):
        super(PackedIntegerEncoder, self).__init__()
        self.encoder = encoder
        self.encoder_epsilon = encoder_epsilon
        self.packed_epsilon = packed_epsilon
        self.shift = shift

    def forward(self, integer_images):
        e2 = self.encoder(integer_images) * self.encoder_epsilon
        e2_integer = torch.clamp(torch.round(e2 / self.packed_epsilon), 0, 255)
        state = state_tensor(self.shift).to(e2)
        state_integer = torch.clamp(torch.round(state / self.packed_epsilon), 0, 255)
        return torch.cat((e2_integer, state_integer), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    parser.add_argument("--signed-weight-bits", type=int, choices=(7, 8), default=7)
    args = parser.parse_args()
    bridge_report = json.loads((args.bridge / "bridge_report.json").read_text())
    if bridge_report["format"] != "dory-motion-gate-ttc-bridge-v1":
        raise RuntimeError("unexpected bridge format")
    samples = json.loads((args.bridge / "samples.json").read_text())
    global SAMPLES
    base.image_tensor = paired_image_tensor

    encoder = EncoderNet()
    base.load_archive(encoder, args.bridge / "encoder_float_state.npz")
    SAMPLES = samples["calibration"]
    calibration = list(range(len(SAMPLES)))
    parity_offset = len(SAMPLES)
    SAMPLES = samples["calibration"] + samples["parity"]
    parity = list(range(parity_offset, len(SAMPLES)))
    encoder_integer, encoder_float, encoder_report = base.quantize_encoder(
        encoder, calibration, parity, args.output / "encoder",
        args.residual_requantization_factor,
        signed_weight_bits=args.signed_weight_bits,
    )

    gate = GateHead()
    load_gate(gate, args.bridge / "gate_head_float_state.npz")
    gate_report = base.quantize_head(
        "gate_head", gate, encoder_integer, encoder_float,
        encoder_report["output_epsilon"], calibration, parity,
        args.output / "gate_head", args.residual_requantization_factor,
        signed_weight_bits=args.signed_weight_bits,
    )

    shift = float(bridge_report["state_shift"])
    packed_epsilon = max(float(encoder_report["output_epsilon"]), (2.0 * shift) / 255.0)
    ttc = TTCHead(int(bridge_report["ttc_refinements"]))
    load_ttc(ttc, args.bridge / "ttc_head_float_state.npz", shift)
    packed_float = PackedFloatEncoder(encoder_float, shift)
    packed_integer = PackedIntegerEncoder(
        encoder_integer, encoder_report["output_epsilon"], packed_epsilon, shift
    )
    ttc_report = base.quantize_head(
        "ttc_head", ttc, packed_integer, packed_float, packed_epsilon,
        calibration, parity, args.output / "ttc_head",
        args.residual_requantization_factor,
        signed_weight_bits=args.signed_weight_bits,
    )
    report = {
        "format": "dory-motion-gate-ttc-nemo-v1",
        "bridge": bridge_report,
        "input_layout": "HWC two-frame uint8 encoder input",
        "ttc_packed_input": {
            "shape": [74, 20, 20],
            "channels": "64 requantized e2 then 10 broadcast normalized-state-plus-4",
            "epsilon": packed_epsilon,
            "state_shift": shift,
        },
        "graphs": [encoder_report, gate_report, ttc_report],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nemo_ttc_gate_dory_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
