#!/usr/bin/env python3
"""Jointly fine-tune ESPNetV2-Lite with middle-tap corner and gate heads."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import MultiTaskDataset
from .espnet_frozen_heads import ESPNetV2LiteEncoder, encoder_fingerprint
from .evaluate import local_centroid
from .losses import soft_dice_loss, weighted_corner_mse
from .model import ConvBNReLU, DSConv


class JointMiddleESPNet(nn.Module):
    def __init__(self, backbone_checkpoint, corner_checkpoint, gate_checkpoint):
        super().__init__()
        saved = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
        self.encoder = ESPNetV2LiteEncoder(2)
        self.encoder.load_state_dict({
            key.removeprefix("encoder."): value
            for key, value in saved["model"].items() if key.startswith("encoder.")
        }, strict=True)
        self.initial_backbone_sha256 = encoder_fingerprint(self.encoder)
        self.corner_adapter = ConvBNReLU(64, 32, 1)
        self.corner_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 4, 1))
        self.gate_adapter = ConvBNReLU(64, 32, 1)
        self.gate_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 1, 1))
        self._load_head(corner_checkpoint, "corner")
        self._load_head(gate_checkpoint, "gate")

    def _load_head(self, checkpoint, prefix):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["head"]
        adapter = {key.removeprefix("adapter."): value for key, value in state.items()
                   if key.startswith("adapter.")}
        head = {key.removeprefix("head."): value for key, value in state.items()
                if key.startswith("head.")}
        getattr(self, f"{prefix}_adapter").load_state_dict(adapter, strict=True)
        getattr(self, f"{prefix}_head").load_state_dict(head, strict=True)

    def forward(self, image):
        # The synthetic captures are independent poses, not temporal sequences.
        # Repeat the current image rather than pairing unrelated frames.
        _, middle, _ = self.encoder(image.repeat(1, 2, 1, 1))
        corners = self.corner_head(self.corner_adapter(middle))
        gate = self.gate_head(self.gate_adapter(middle))
        return {
            "corners": F.interpolate(corners, (40, 40), mode="bilinear", align_corners=False),
            "gate": F.interpolate(gate, (40, 40), mode="bilinear", align_corners=False),
        }


def run_epoch(model, loader, device, corner_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "corner_loss", "gate_bce", "gate_dice",
                                    "intersection", "union", "predicted", "truth")}
    samples, errors, within5, within10 = 0, [], [], []
    with (torch.enable_grad() if training else torch.no_grad()):
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            corner_maps = batch["corners"].to(device, non_blocking=True)
            corner_xy = batch["corner_xy"].to(device, non_blocking=True)
            valid = batch["corner_valid"].to(device, non_blocking=True)
            gate_target = batch["gate"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(image)
            corner_loss = weighted_corner_mse(output["corners"], corner_maps, valid)
            gate_bce = F.binary_cross_entropy_with_logits(output["gate"], gate_target)
            gate_dice = soft_dice_loss(output["gate"], gate_target)
            loss = corner_weight * corner_loss + gate_bce + 0.5 * gate_dice
            if training:
                loss.backward()
                optimizer.step()
            size = image.shape[0]
            samples += size
            for key, value in (("loss", loss), ("corner_loss", corner_loss),
                               ("gate_bce", gate_bce), ("gate_dice", gate_dice)):
                totals[key] += float(value.detach()) * size
            prediction = local_centroid(output["corners"].sigmoid()) * 4.0
            if valid.any():
                error = torch.linalg.vector_norm(prediction[valid] - corner_xy[valid], dim=2)
                errors.append(error.detach().cpu())
                within5.append((error <= 5).all(1).detach().cpu())
                within10.append((error <= 10).all(1).detach().cpu())
            predicted_gate = output["gate"].sigmoid() >= 0.5
            true_gate = gate_target >= 0.5
            totals["intersection"] += float((predicted_gate & true_gate).sum())
            totals["union"] += float((predicted_gate | true_gate).sum())
            totals["predicted"] += float(predicted_gate.sum())
            totals["truth"] += float(true_gate.sum())
    error = torch.cat(errors) if errors else torch.empty(0, 4)
    intersection = totals["intersection"]
    return {
        "loss": totals["loss"] / samples,
        "corner_loss": totals["corner_loss"] / samples,
        "mean_corner_error_px": float(error.mean()) if error.numel() else float("nan"),
        "median_corner_error_px": float(error.median()) if error.numel() else float("nan"),
        "p95_corner_error_px": float(torch.quantile(error, 0.95)) if error.numel() else float("nan"),
        "all4_within_5px": float(torch.cat(within5).float().mean()) if within5 else float("nan"),
        "all4_within_10px": float(torch.cat(within10).float().mean()) if within10 else float("nan"),
        "gate_bce": totals["gate_bce"] / samples,
        "gate_soft_dice_loss": totals["gate_dice"] / samples,
        "gate_iou": intersection / max(1.0, totals["union"]),
        "gate_f1": 2 * intersection / max(1.0, totals["predicted"] + totals["truth"]),
        "gate_precision": intersection / max(1.0, totals["predicted"]),
        "gate_recall": intersection / max(1.0, totals["truth"]),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("backbone-checkpoint", "corner-checkpoint", "gate-checkpoint",
                 "dataset", "targets", "split-file", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=2e-3)
    parser.add_argument("--corner-weight", type=float, default=10.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {split: MultiTaskDataset(args.dataset, args.targets, args.split_file, split)
                for split in ("train", "validation")}
    loaders = {split: DataLoader(dataset, args.batch_size, shuffle=split == "train",
                                  num_workers=args.workers, pin_memory=device.type == "cuda",
                                  persistent_workers=args.workers > 0)
               for split, dataset in datasets.items()}
    model = JointMiddleESPNet(args.backbone_checkpoint, args.corner_checkpoint,
                              args.gate_checkpoint).to(device)
    head_parameters = list(model.corner_adapter.parameters()) + list(model.corner_head.parameters())
    head_parameters += list(model.gate_adapter.parameters()) + list(model.gate_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.backbone_learning_rate},
        {"params": head_parameters, "lr": args.head_learning_rate},
    ], weight_decay=1e-4)
    best_score = float("inf")
    with (args.output / "log.csv").open("w", newline="") as stream:
        writer = None
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            train = run_epoch(model, loaders["train"], device, args.corner_weight, optimizer)
            validation = run_epoch(model, loaders["validation"], device, args.corner_weight)
            row = {"epoch": epoch, "seconds": time.time() - started,
                   **{f"train_{key}": value for key, value in train.items()},
                   **{f"val_{key}": value for key, value in validation.items()}}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); stream.flush()
            print(json.dumps({"epoch": epoch, "train": train, "validation": validation,
                              "backbone_changed": encoder_fingerprint(model.encoder) != model.initial_backbone_sha256}), flush=True)
            # Lower is better; the normalized IoU penalty keeps model selection joint.
            score = validation["mean_corner_error_px"] + 10.0 * (1.0 - validation["gate_iou"])
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "validation": validation, "joint_selection_score": score,
                     "initial_backbone_sha256": model.initial_backbone_sha256,
                     "final_backbone_sha256": encoder_fingerprint(model.encoder),
                     "backbone_finetuned": True, "tap": "middle", "corner_head": "heatmap",
                     "gate_head": "binary", "temporal_input_policy": "repeat_current_frame"}
            torch.save(state, args.output / "last.pt")
            if score < best_score:
                best_score = score; torch.save(state, args.output / "best.pt")
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    (args.output / "summary.json").write_text(json.dumps({
        "best_epoch": best["epoch"], "validation": best["validation"],
        "joint_selection_score": best["joint_selection_score"], "tap": "middle",
        "corner_head": "heatmap", "gate_head": "binary", "backbone_finetuned": True,
        "initial_backbone_sha256": best["initial_backbone_sha256"],
        "final_backbone_sha256": best["final_backbone_sha256"],
        "corner_weight": args.corner_weight, "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate, "temporal_input_policy": "repeat_current_frame",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
