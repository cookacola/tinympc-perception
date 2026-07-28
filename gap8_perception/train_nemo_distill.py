#!/usr/bin/env python
"""NEMO-native 8-bit distillation for the DORY deployment graph.

This file intentionally stays Python-3.7 compatible because the cluster's
working NEMO environment is pinned to that interpreter.
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

from gap8_perception.nemo_integer_export import PackedDoryNet


def _autograd_safe_integer_add(self, *inputs):
    """Training-only replacement for old NEMO's in-place residual add."""
    output = inputs[0].clone()
    for value in inputs[1:]:
        output = output + value
    return output


nemo.quant.pact.PACT_IntegerAdd.forward = _autograd_safe_integer_add


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(image).unsqueeze(0).float() / 255.0


def load_bridge(model, bridge):
    archive = np.load(str(bridge / "packed_float_state.npz"))
    state = model.state_dict()
    shared = {key for key in archive.files if not key.startswith("packed_head.1.")}
    state.update({key: torch.from_numpy(archive[key]) for key in shared})
    state["output_proj.0.weight"] = torch.from_numpy(
        archive["packed_head.1.weight"]
    )
    eps = model.output_proj[1].eps
    state["output_proj.1.weight"] = torch.full_like(
        state["output_proj.1.weight"], (1.0 + eps) ** 0.5
    )
    state["output_proj.1.bias"] = torch.from_numpy(
        archive["packed_head.1.bias"]
    )
    state["output_proj.1.running_mean"].zero_()
    state["output_proj.1.running_var"].fill_(1.0)
    model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = json.loads((args.bridge / "bridge_report.json").read_text())
    base = PackedDoryNet(report["packed_channels"]).eval()
    load_bridge(base, args.bridge)
    teacher = copy.deepcopy(base).eval()

    paths = sorted(args.dataset.glob("shard_*/hm01b0_mono_*.png"))
    if not paths:
        raise RuntimeError("no HM01B0 images")
    calibration = [
        paths[index] for index in np.linspace(
            0, len(paths) - 1, min(args.calibration_images, len(paths))
        ).astype(int)
    ]

    spatial_min = np.full(report["packed_channels"], np.inf, np.float64)
    learned_bias = (
        teacher.output_proj[1].bias.detach().cpu().numpy().copy()
    )
    with torch.no_grad():
        for start in range(0, len(calibration), args.batch_size):
            images = Images(calibration[start:start + args.batch_size])
            tensor = torch.stack([images[i] for i in range(len(images))])
            spatial = teacher.output_proj[0](teacher.forward_features(tensor))
            flat = spatial.permute(1, 0, 2, 3).contiguous().view(
                spatial.shape[1], -1
            )
            spatial_min = np.minimum(
                spatial_min, flat.min(dim=1)[0].cpu().numpy()
            )
    output_offset = np.maximum(-spatial_min, 0.0) + 1.0e-4
    with torch.no_grad():
        base.output_proj[1].bias.copy_(
            torch.from_numpy(output_offset).to(base.output_proj[1].bias)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    student = nemo.transform.quantize_pact(base)
    student.change_precision(bits=8)
    student.reset_alpha_weights()
    student.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration), args.batch_size):
            images = Images(calibration[start:start + args.batch_size])
            student(
                torch.stack([images[i] for i in range(len(images))]).to(device)
            )
    student.unset_statistics_act()
    student.reset_alpha_act()

    teacher.to(device)
    student.to(device)
    offset = torch.tensor(
        output_offset, dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    bias = torch.tensor(
        learned_bias, dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    loader = DataLoader(
        Images(paths), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.learning_rate
    )
    start_epoch = 1
    if args.resume:
        resume = torch.load(str(args.resume), map_location=device)
        if resume.get("architecture") != report["architecture"]:
            raise RuntimeError("resume architecture mismatch")
        if not np.allclose(
            np.asarray(resume["output_offset"]), output_offset,
            rtol=1.0e-5, atol=1.0e-5,
        ):
            raise RuntimeError("resume output offset mismatch")
        student.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"]) + 1
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        # Keep the trained float BatchNorm statistics fixed. Only weights and
        # PACT clipping parameters adapt to quantization.
        student.eval()
        total = 0.0
        items = 0
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                target = teacher.forward_logits(images)
            physical = student(images)
            decoded = physical - offset + bias
            error = decoded - target
            loss = error.pow(2).mean() + 0.02 * error.abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * images.shape[0]
            items += images.shape[0]
        value = total / items
        history.append({"epoch": epoch, "distillation_loss": value})
        checkpoint = {
            "epoch": epoch,
            "model": student.state_dict(),
            "architecture": report["architecture"],
            "output_offset": output_offset.tolist(),
            "learned_bias": learned_bias.tolist(),
            "precision_bits": 8,
        }
        for checkpoint_path in (
            args.output / "last_nemo_qat.pt",
            args.output / ("epoch_%02d_nemo_qat.pt" % epoch),
        ):
            torch.save(
                checkpoint, str(checkpoint_path),
                _use_new_zipfile_serialization=False,
            )
        print(json.dumps(history[-1]), flush=True)
    (args.output / "nemo_qat_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
