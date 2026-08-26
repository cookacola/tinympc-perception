#!/usr/bin/env python3
"""Train a DroNet-style obstacle branch on the published ESPNet student.

The shared gate stem and stage-1 feature extractor stay frozen, including
BatchNorm statistics.  Only the obstacle-only deep stages and a global
pooling/96-to-2 navigation head are optimized on the official PULP-DroNet
yaw-rate and collision labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PULP_ROOT = Path("/home/cchen/pulp-dronet/tiny-pulp-dronet-v3")
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PULP_ROOT) not in sys.path:
    sys.path.insert(0, str(PULP_ROOT))

from fair_espnet_dronetv3_benchmark import (  # noqa: E402
    evaluate,
    find_strict_pairs,
    make_loader,
    profile_model,
    sample_fingerprint,
    seed_everything,
    select_f1_threshold,
    train_epoch,
)

from gap8_perception.model_espnet_dory_student import ESPNetDoryStudent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ESPNetDroNetNavigation(nn.Module):
    """Published ESPNet gate trunk plus DroNet's two scalar navigation outputs."""

    input_shape = (2, 160, 160)

    def __init__(self, checkpoint: Path):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("architecture") != "ESPNetDoryStudent":
            raise ValueError(f"unexpected architecture in {checkpoint}")
        source = ESPNetDoryStudent()
        source.load_state_dict(saved["model"], strict=True)

        self.stem = source.stem
        self.stage1 = source.stage1
        self.corner_head = source.corner_head
        self.gate_head = source.gate_head
        self.stage2 = source.stage2
        self.stage3 = source.stage3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.navigation_head = nn.Linear(96, 2, bias=False)

        self._frozen_modules = (
            self.stem,
            self.stage1,
            self.corner_head,
            self.gate_head,
        )
        for module in self._frozen_modules:
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen_modules:
            module.eval()
        return self

    def shared_features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(frames))

    def forward_gate(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_features(frames)
        return {
            "corners": self.corner_head(shared),
            "gate": self.gate_head(shared),
        }

    def forward(self, frames: torch.Tensor):
        shared = self.shared_features(frames)
        deep = self.stage3(self.stage2(shared))
        outputs = self.navigation_head(self.dropout(self.pool(deep).flatten(1)))
        return [outputs[:, 0], torch.sigmoid(outputs[:, 1])]


@torch.no_grad()
def gate_preservation_error(model: ESPNetDroNetNavigation, checkpoint: Path) -> float:
    reference = ESPNetDoryStudent().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference.load_state_dict(saved["model"], strict=True)
    generator = torch.Generator().manual_seed(18082026)
    frames = torch.rand((4, 2, 160, 160), generator=generator)
    expected = reference(frames)
    actual = model.cpu().eval().forward_gate(frames)
    return max(
        float((actual[name] - expected[name]).abs().max())
        for name in ("corners", "gate")
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full training requires a Slurm-allocated CUDA GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda")

    samples = {
        split: find_strict_pairs(args.data, split)
        for split in ("train", "valid", "test")
    }
    current_sets = {
        split: {sample.current for sample in values}
        for split, values in samples.items()
    }
    overlap = {
        f"{left}_{right}": len(current_sets[left] & current_sets[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    manifest = {
        "strict_definition": (
            "Adjacent numeric JPEG ranks, same labels_partitioned.csv acquisition, "
            "and both rows in the requested official partition"
        ),
        "labels_from": "current frame",
        "counts": {split: len(values) for split, values in samples.items()},
        "fingerprints_sha256": {
            split: sample_fingerprint(values) for split, values in samples.items()
        },
        "current_frame_overlap": overlap,
    }
    reference_manifest = json.loads(args.reference_manifest.read_text())
    for key in ("counts", "fingerprints_sha256", "current_frame_overlap"):
        if manifest[key] != reference_manifest[key]:
            raise RuntimeError(f"official benchmark manifest mismatch for {key}")
    (args.output / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    loaders = {
        split: make_loader(
            values,
            frame_count=2,
            augment=split == "train",
            args=args,
            shuffle=split == "train",
        )
        for split, values in samples.items()
    }
    model = ESPNetDroNetNavigation(args.base_checkpoint)
    nn.init.xavier_uniform_(model.navigation_head.weight)
    profile = profile_model(model, 2)
    active_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_loss = math.inf
    remaining_patience = args.patience
    history = []
    best_path = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, loaders["train"], device, optimizer)
        validation, _, _ = evaluate(model, loaders["valid"], device)
        selection_loss = validation["yaw_mse"] + validation["collision_bce"]
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
            "selection_loss": selection_loss,
        })
        print(
            f"epoch {epoch:03d}: loss={selection_loss:.6f}, "
            f"yaw_rmse={validation['yaw_rmse']:.4f}, "
            f"collision_bce={validation['collision_bce']:.4f}, "
            f"AUROC={validation['collision_auroc']:.4f}",
            flush=True,
        )
        if selection_loss < best_loss - args.min_delta:
            best_loss = selection_loss
            remaining_patience = args.patience
            torch.save({
                "format": "espnet-dronet-navigation-v1",
                "epoch": epoch,
                "model": model.state_dict(),
                "base_checkpoint_sha256": sha256(args.base_checkpoint),
                "frozen_gate_trunk": True,
            }, best_path)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    validation, validation_truth, validation_scores = evaluate(
        model, loaders["valid"], device
    )
    threshold = select_f1_threshold(validation_truth, validation_scores)
    test, _, _ = evaluate(model, loaders["test"], device, threshold)
    gate_error = gate_preservation_error(model, args.base_checkpoint)
    if gate_error != 0.0:
        raise RuntimeError(f"frozen gate output changed by {gate_error}")

    result = {
        "name": "published_espnet_dronet_navigation",
        "seed": args.seed,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "selected_epoch": selected["epoch"],
        "epochs_trained": len(history),
        "best_validation_loss": best_loss,
        "validation_selected_f1_threshold": threshold,
        "output_contract": {
            "steering": "normalized yaw rate",
            "collision": "sigmoid probability",
        },
        "frozen_gate_trunk": True,
        "gate_preservation_max_abs_error": gate_error,
        "trainable_parameters": active_parameters,
        **profile,
        "validation": validation,
        "test": test,
        "weights": str(best_path),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
