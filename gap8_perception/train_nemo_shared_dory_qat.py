#!/usr/bin/env python
"""QAT fine tuning for the three independently generated shared-STDC graphs.

The float teacher and fake-quant student see identical encoder tensors.  This
keeps the encoder/head ABI explicit while allowing PACT activation ranges at
residual joins to adapt before integer export.
"""

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

from gap8_perception.nemo_stdc_shared_dory_export import (
    CornerHeadNet, DangerHeadNet, EncoderNet, image_tensor, load_archive,
    load_head,
)


def _autograd_safe_integer_add(self, *inputs):
    output = inputs[0].clone()
    for value in inputs[1:]:
        output = output + value
    return output


# NeMO 0.0.x uses an in-place residual addition that invalidates autograd in
# modern PyTorch.  The integer deployment arithmetic is unchanged; this only
# makes fake-quant fine tuning differentiable.
nemo.quant.pact.PACT_IntegerAdd.forward = _autograd_safe_integer_add


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(image[20:140]).unsqueeze(0).float() / 255.0


def configure_adds(model, factor):
    for module in model.modules():
        if hasattr(module, "requantization_factor"):
            module.requantization_factor = factor


def build(graph, bridge):
    encoder = EncoderNet()
    load_archive(encoder, bridge / "encoder_float_state.npz")
    if graph == "encoder":
        return encoder, None, (1, 120, 160)
    model = CornerHeadNet() if graph == "corner_head" else DangerHeadNet()
    load_head(model, bridge / (graph + "_float_state.npz"), graph)
    return model, encoder.eval(), model.input_shape


def calibration_input(graph, images, encoder):
    if graph == "encoder":
        return images
    with torch.no_grad():
        return encoder(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=("encoder", "corner_head", "danger_head"), required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--residual-requantization-factor", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.dataset.glob("shard_*/hm01b0_mono_*.png"))
    if args.train_limit:
        paths = paths[:args.train_limit]
    if not paths:
        raise RuntimeError("no HM01B0 training images")
    base, encoder, shape = build(args.graph, args.bridge)
    teacher = copy.deepcopy(base).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if encoder is not None:
        encoder = encoder.to(device).eval()
    student = nemo.transform.quantize_pact(
        base, dummy_input=torch.ones(1, *shape)
    )
    student.change_precision(bits=8)
    student.reset_alpha_weights()
    student.to(device)
    student.set_statistics_act()
    calibration_indices = np.linspace(
        0, len(paths) - 1, min(args.calibration_images, len(paths))
    ).astype(int)
    with torch.no_grad():
        for start in range(0, len(calibration_indices), args.batch_size):
            batch_paths = [paths[i] for i in calibration_indices[start:start + args.batch_size]]
            images = torch.stack([Images(batch_paths)[i] for i in range(len(batch_paths))]).to(device)
            student(calibration_input(args.graph, images, encoder))
    student.unset_statistics_act()
    student.reset_alpha_act()
    configure_adds(student, args.residual_requantization_factor)

    teacher, student = teacher.to(device), student.to(device)
    # The pinned Frontnet environment uses an older PyTorch release whose
    # DataLoader predates ``persistent_workers``.
    loader = DataLoader(Images(paths), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        total, items = 0.0, 0
        for images in loader:
            images = images.to(device, non_blocking=True)
            source = calibration_input(args.graph, images, encoder)
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
            items += images.shape[0]
        metric = {"epoch": epoch, "distillation_loss": total / max(items, 1)}
        history.append(metric)
        checkpoint = {
            "graph": args.graph,
            "model": student.state_dict(),
            "precision_bits": 8,
            "residual_requantization_factor": args.residual_requantization_factor,
            "source_bridge": str(args.bridge),
        }
        torch.save(checkpoint, str(args.output / (args.graph + "_qat.pt")),
                   _use_new_zipfile_serialization=False)
        print(json.dumps(metric), flush=True)
    (args.output / (args.graph + "_qat_history.json")).write_text(
        json.dumps(history, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
