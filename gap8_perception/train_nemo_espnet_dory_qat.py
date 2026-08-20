#!/usr/bin/env python
"""NEMO QAT distillation for the four two-frame deployment graphs."""

from __future__ import print_function

import argparse
import copy
import json
from pathlib import Path

import cv2
import nemo
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gap8_perception.nemo_espnet_dory_student_export import (
    DangerHeadNet,
    EncoderNet,
    FineHeadNet,
    temporal_pairs,
)
from gap8_perception.nemo_stdc_shared_dory_export import load_archive, load_head


def safe_integer_add(self, *inputs):
    output = inputs[0].clone()
    for value in inputs[1:]:
        output = output + value
    return output


nemo.quant.pact.PACT_IntegerAdd.forward = safe_integer_add


class PairImages(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        frames = []
        for path in self.pairs[index]:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (160, 160):
                raise RuntimeError("invalid HM01B0 frame %s" % path)
            frames.append(image)
        return torch.from_numpy(np.stack(frames)).float() / 255.0


def configure_adds(model, factor):
    for module in model.modules():
        if hasattr(module, "requantization_factor"):
            module.requantization_factor = factor


def build(graph, bridge):
    architecture = json.loads(
        (bridge / "bridge_report.json").read_text()
    )["architecture"]
    encoder = EncoderNet(architecture)
    load_archive(encoder, bridge / "encoder_float_state.npz")
    if graph == "encoder":
        return encoder, None, encoder.input_shape
    if graph == "corner_head":
        model = FineHeadNet(4)
    elif graph == "gate_head":
        model = FineHeadNet(1)
    else:
        model = DangerHeadNet(architecture)
    load_head(model, bridge / (graph + "_float_state.npz"), graph)
    return model, encoder.eval(), model.input_shape


def source_tensor(graph, images, encoder):
    if graph == "encoder":
        return images
    with torch.no_grad():
        return encoder(images)


def shift_terminal_activation(model, encoder, calibration, batch_size):
    spatial_min = np.full(model.output_channels, np.inf, np.float64)
    with torch.no_grad():
        for start in range(0, len(calibration), batch_size):
            images = torch.stack([
                PairImages(calibration[start:start + batch_size])[index]
                for index in range(len(calibration[start:start + batch_size]))
            ])
            shared = encoder(images)
            spatial = model.output_proj[0](model.forward_features(shared))
            flat = spatial.permute(1, 0, 2, 3).contiguous().view(
                model.output_channels, -1
            )
            spatial_min = np.minimum(spatial_min, flat.min(1)[0].numpy())
    offset = np.maximum(-spatial_min, 0.0) + 1.0e-4
    with torch.no_grad():
        model.output_proj[1].bias.copy_(
            torch.from_numpy(offset).to(model.output_proj[1].bias)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        choices=("encoder", "corner_head", "gate_head", "danger_head"),
        required=True,
    )
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = temporal_pairs(args.dataset, "train")
    if args.train_limit and len(pairs) > args.train_limit:
        indices = np.linspace(0, len(pairs) - 1, args.train_limit).astype(int)
        pairs = [pairs[index] for index in indices]
    calibration = pairs[:min(args.calibration_images, len(pairs))]
    base, encoder, shape = build(args.graph, args.bridge)
    if encoder is not None:
        shift_terminal_activation(base, encoder, calibration, args.batch_size)
    teacher = copy.deepcopy(base).eval()
    student = nemo.transform.quantize_pact(
        base, dummy_input=torch.ones(1, *shape)
    )
    student.change_precision(bits=8)
    student.reset_alpha_weights()
    student.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration), args.batch_size):
            images = torch.stack([
                PairImages(calibration[start:start + args.batch_size])[index]
                for index in range(len(calibration[start:start + args.batch_size]))
            ])
            student(source_tensor(args.graph, images, encoder))
    student.unset_statistics_act()
    student.reset_alpha_act()
    configure_adds(student, args.residual_requantization_factor)

    loader = DataLoader(
        PairImages(pairs), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0.0
        examples = 0
        for images in loader:
            source = source_tensor(args.graph, images, encoder)
            with torch.no_grad():
                target = teacher(source)
            prediction = student(source)
            error = prediction - target
            loss = error.pow(2).mean() + 0.02 * error.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * images.shape[0]
            examples += images.shape[0]
        metric = {"epoch": epoch, "distillation_loss": total / examples}
        history.append(metric)
        checkpoint = {
            "graph": args.graph,
            "model": student.state_dict(),
            "precision_bits": 8,
            "residual_requantization_factor": args.residual_requantization_factor,
            "source_bridge": str(args.bridge),
        }
        torch.save(
            checkpoint, str(args.output / (args.graph + "_qat.pt")),
            _use_new_zipfile_serialization=False,
        )
        print(json.dumps(metric), flush=True)
    (args.output / (args.graph + "_qat_history.json")).write_text(
        json.dumps(history, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
