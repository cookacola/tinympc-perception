#!/usr/bin/env python3
"""Train the deployed gate heads at three frozen PULP-DroNetV3 taps."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from gap8_perception.data import MultiTaskDataset
from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.evaluate import local_centroid
from gap8_perception.losses import soft_dice_loss, weighted_corner_mse
from gap8_perception.model import ConvBNReLU, DSConv


TAP_CHANNELS = {"block1": 32, "block2": 64, "block3": 128}


class NoGateDataset(Dataset):
    def __init__(self, root: Path, shard_indices):
        self.paths = []
        for index in shard_indices:
            shard = root / f"shard_{index * 1000:09d}"
            if not (shard / "_SUCCESS").is_file():
                raise FileNotFoundError(f"incomplete no-gate shard: {shard}")
            self.paths.extend(sorted(shard.glob("hm01b0_mono_*.png")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(path)
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
            "source": str(path),
        }


class RealGateDataset(Dataset):
    def __init__(self, root: Path, flights):
        self.records = []
        for flight in flights:
            folder = root / flight
            for line in (folder / "labels.jsonl").read_text().splitlines():
                row = json.loads(line)
                corners = canonical_image_order(
                    np.asarray(row["corners"], np.float32).reshape(4, 2)
                )[0]
                if (
                    (corners < 0).any()
                    or (corners[:, 0] >= 160).any()
                    or (corners[:, 1] >= 160).any()
                ):
                    continue
                self.records.append(
                    (folder / "stream_out" / row["image"], corners, flight)
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, corners, flight = self.records[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(path)
        yy, xx = np.mgrid[:40, :40]
        maps = np.zeros((4, 40, 40), np.float32)
        for channel, (x, y) in enumerate(corners / 4.0):
            maps[channel] = np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.25**2)
            )
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
            "corners": torch.from_numpy(maps),
            "corner_xy": torch.from_numpy(corners.copy()),
            "corner_valid": torch.tensor(True),
            "flight": flight,
            "source": str(path),
        }


def fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenDroNetV3GateHeads(nn.Module):
    """Official two-frame DroNetV3 plus matched heatmap/mask heads."""

    def __init__(self, checkpoint: Path, pulp_root: Path, tap: str):
        super().__init__()
        if tap not in TAP_CHANNELS:
            raise ValueError(tap)
        sys.path.insert(0, str(pulp_root))
        from model.dronet_v3 import Depthwise_Separable, dronet

        self.backbone = dronet(
            depth_mult=1.0,
            block_class=Depthwise_Separable,
            bypass=False,
            input_channels=2,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.keep_backbone_frozen = True
        self.backbone_fingerprint = fingerprint(self.backbone)
        self.tap = tap

        channels = TAP_CHANNELS[tap]
        self.corner_adapter = ConvBNReLU(channels, 32, 1)
        self.corner_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 4, 1))
        self.gate_adapter = ConvBNReLU(channels, 32, 1)
        self.gate_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 1, 1))
        self.presence_head = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.keep_backbone_frozen:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self):
        self.keep_backbone_frozen = False
        self.backbone.requires_grad_(True)
        return self

    def _features(self, image: torch.Tensor):
        # The published checkpoint was trained at 200x200. Resizing preserves
        # that operating point while outputs remain expressed in the calibrated
        # 160x160 camera coordinate system.
        frames = F.interpolate(
            image.repeat(1, 2, 1, 1),
            size=(200, 200),
            mode="bilinear",
            align_corners=False,
        )
        out = self.backbone.relu1(self.backbone.bn1(self.backbone.first_conv(frames)))
        out = self.backbone.pool(out)
        block1 = self.backbone.Block1(out)
        block2 = self.backbone.Block2(block1)
        block3 = self.backbone.Block3(block2)
        return block1, block2, block3

    def forward_gate(self, image: torch.Tensor):
        was_training = self.backbone.training
        self.backbone.eval()
        gradient_context = (
            torch.no_grad() if self.keep_backbone_frozen else contextlib.nullcontext()
        )
        with gradient_context:
            features = self._features(image)
        self.backbone.train(was_training)
        feature = features[{"block1": 0, "block2": 1, "block3": 2}[self.tap]]
        corner_feature = self.corner_adapter(feature)
        gate_feature = self.gate_adapter(feature)
        corners = self.corner_head(corner_feature)
        gate = self.gate_head(gate_feature)
        presence = self.presence_head(
            F.adaptive_avg_pool2d(gate_feature, 1).flatten(1)
        ).squeeze(1)
        return {
            "corners": F.interpolate(
                corners, (40, 40), mode="bilinear", align_corners=False
            ),
            "gate": F.interpolate(
                gate, (40, 40), mode="bilinear", align_corners=False
            ),
            "presence_logit": presence,
        }

    def forward_navigation(self, frames: torch.Tensor):
        if frames.shape[1:] != (2, 200, 200):
            raise ValueError(f"expected Bx2x200x200 frames, got {tuple(frames.shape)}")
        out = self.backbone.relu1(
            self.backbone.bn1(self.backbone.first_conv(frames))
        )
        out = self.backbone.pool(out)
        out = self.backbone.Block3(
            self.backbone.Block2(self.backbone.Block1(out))
        )
        if not self.backbone.nemo:
            out = self.backbone.dropout(out)
        output = self.backbone.fc(out.flatten(1))
        return [output[:, 0], torch.sigmoid(output[:, 1])]

    def forward_combined(self, image: torch.Tensor):
        """Run navigation and gate paths once for parameter/MAC profiling."""
        features = self._features(image)
        navigation = self.backbone.fc(features[2].flatten(1))
        feature = features[{"block1": 0, "block2": 1, "block3": 2}[self.tap]]
        corner_feature = self.corner_adapter(feature)
        gate_feature = self.gate_adapter(feature)
        corners = self.corner_head(corner_feature)
        gate = self.gate_head(gate_feature)
        presence = self.presence_head(
            F.adaptive_avg_pool2d(gate_feature, 1).flatten(1)
        )
        return navigation, corners, gate, presence

    def assert_backbone_unchanged(self) -> str:
        current = fingerprint(self.backbone)
        if current != self.backbone_fingerprint:
            raise RuntimeError("frozen DroNetV3 backbone changed")
        return current


def limited(dataset, limit: int):
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return Subset(dataset, range(limit))


def infinite(loader):
    while True:
        yield from loader


def gate_loss(model, synthetic, real, negative, device):
    synthetic_image = synthetic["image"].to(device, non_blocking=True)
    target_mask = synthetic["gate"].to(device, non_blocking=True)
    target_corners = synthetic["corners"].to(device, non_blocking=True)
    valid = synthetic["corner_valid"].to(device, non_blocking=True)
    output = model.forward_gate(synthetic_image)
    corner = weighted_corner_mse(output["corners"], target_corners, valid)
    mask = F.binary_cross_entropy_with_logits(output["gate"], target_mask)
    mask = mask + 0.5 * soft_dice_loss(output["gate"], target_mask)

    negative_output = model.forward_gate(
        negative["image"].to(device, non_blocking=True)
    )
    negative_mask = F.binary_cross_entropy_with_logits(
        negative_output["gate"], torch.zeros_like(negative_output["gate"])
    )
    present = target_mask.flatten(1).any(1).float()
    confidence = F.binary_cross_entropy_with_logits(
        torch.cat((output["presence_logit"], negative_output["presence_logit"])),
        torch.cat((present, torch.zeros_like(negative_output["presence_logit"]))),
    )

    real_output = model.forward_gate(real["image"].to(device, non_blocking=True))
    real_corner = weighted_corner_mse(
        real_output["corners"],
        real["corners"].to(device, non_blocking=True),
        real["corner_valid"].to(device, non_blocking=True),
    )
    real_presence = F.binary_cross_entropy_with_logits(
        real_output["presence_logit"], torch.ones_like(real_output["presence_logit"])
    )
    total = (
        10.0 * corner
        + mask
        + confidence
        + 5.0 * real_corner
        + 0.25 * real_presence
        + negative_mask
    )
    return total, {
        "synthetic_corner": corner,
        "synthetic_mask": mask,
        "confidence": confidence,
        "real_corner": real_corner,
        "real_presence": real_presence,
        "negative_mask": negative_mask,
    }


def train_epoch(model, loaders, optimizer, device):
    model.train()
    streams = {name: infinite(loader) for name, loader in loaders.items()}
    steps = max(len(loader) for loader in loaders.values())
    totals = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = gate_loss(
            model,
            next(streams["synthetic"]),
            next(streams["real"]),
            next(streams["negative"]),
            device,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite gate loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            10.0,
        )
        optimizer.step()
        values = {"total": loss, **parts}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / steps for name, value in totals.items()}


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_labels = labels[order]
    if not sorted_labels.any():
        return float("nan")
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels].mean())


@torch.no_grad()
def evaluate(model, synthetic_loader, negative_loader, real_loader, device):
    model.eval()
    intersection = union = predicted_pixels = truth_pixels = 0.0
    corner_errors = []
    real_errors = []
    labels = []
    scores = []
    negative_pixels = negative_total = 0
    real_scores = []

    for batch in synthetic_loader:
        output = model.forward_gate(batch["image"].to(device, non_blocking=True))
        target = batch["gate"].to(device, non_blocking=True) >= 0.5
        prediction = output["gate"].sigmoid() >= 0.5
        intersection += float((prediction & target).sum())
        union += float((prediction | target).sum())
        predicted_pixels += float(prediction.sum())
        truth_pixels += float(target.sum())
        points = local_centroid(output["corners"].sigmoid()) * 4.0
        valid = batch["corner_valid"].to(device, non_blocking=True)
        if valid.any():
            corner_errors.append(
                torch.linalg.vector_norm(
                    points[valid]
                    - batch["corner_xy"].to(device, non_blocking=True)[valid],
                    dim=2,
                ).cpu()
            )
        present = target.flatten(1).any(1)
        labels.extend(present.cpu().tolist())
        scores.extend(output["presence_logit"].sigmoid().cpu().tolist())

    for batch in negative_loader:
        output = model.forward_gate(batch["image"].to(device, non_blocking=True))
        prediction = output["gate"].sigmoid() >= 0.5
        negative_pixels += int(prediction.sum())
        negative_total += prediction.numel()
        probability = output["presence_logit"].sigmoid()
        labels.extend([False] * len(probability))
        scores.extend(probability.cpu().tolist())

    for batch in real_loader:
        output = model.forward_gate(batch["image"].to(device, non_blocking=True))
        points = local_centroid(output["corners"].sigmoid()) * 4.0
        real_errors.append(
            torch.linalg.vector_norm(
                points - batch["corner_xy"].to(device, non_blocking=True), dim=2
            ).cpu()
        )
        real_scores.extend(output["presence_logit"].sigmoid().cpu().tolist())

    corner = torch.cat(corner_errors)
    real = torch.cat(real_errors)
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    predictions = scores_array >= 0.5
    return {
        "synthetic_gate_iou": intersection / max(1.0, union),
        "synthetic_gate_f1": 2.0 * intersection
        / max(1.0, predicted_pixels + truth_pixels),
        "synthetic_corner_mean_px": float(corner.mean()),
        "synthetic_corner_median_px": float(corner.median()),
        "synthetic_corner_p95_px": float(torch.quantile(corner, 0.95)),
        "synthetic_all4_within_10px": float((corner <= 10).all(1).float().mean()),
        "presence_bce": float(
            F.binary_cross_entropy(
                torch.from_numpy(scores_array),
                torch.from_numpy(labels_array.astype(np.float64)),
            )
        ),
        "presence_auroc": binary_auroc(labels_array, scores_array),
        "presence_ap": average_precision(labels_array, scores_array),
        "presence_accuracy_at_0_5": float((predictions == labels_array).mean()),
        "explicit_no_gate_false_positive_rate_at_0_5": float(
            predictions[~labels_array].mean()
        ),
        "explicit_no_gate_mask_pixel_rate": negative_pixels / max(1, negative_total),
        "real_corner_mean_px": float(real.mean()),
        "real_corner_median_px": float(real.median()),
        "real_corner_p95_px": float(torch.quantile(real, 0.95)),
        "real_all4_within_10px": float((real <= 10).all(1).float().mean()),
        "real_presence_recall_at_0_5": float(
            (np.asarray(real_scores) >= 0.5).mean()
        ),
        "real_examples": len(real_scores),
    }


def profile_combined(model, device):
    macs = 0
    hooks = []

    def convolution_hook(module, inputs, output):
        nonlocal macs
        height, width = output.shape[-2:]
        kernel_height, kernel_width = module.kernel_size
        macs += (
            output.shape[1]
            * height
            * width
            * (module.in_channels // module.groups)
            * kernel_height
            * kernel_width
        )

    def linear_hook(module, inputs, output):
        nonlocal macs
        macs += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model.forward_combined(torch.zeros(1, 1, 160, 160, device=device))
    for hook in hooks:
        hook.remove()
    return {
        "combined_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_parameters": sum(
            parameter.numel() for parameter in model.backbone.parameters()
        ),
        "trainable_gate_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "combined_macs": macs,
        "input": [1, 1, 160, 160],
        "backbone_resize": [200, 200],
    }


def make_datasets(arguments):
    synthetic = {
        split: MultiTaskDataset(
            arguments.gate_dataset,
            arguments.gate_targets,
            arguments.gate_split_file,
            split,
            augment=split == "train",
        )
        for split in ("train", "validation", "test")
    }
    negative = {
        "train": NoGateDataset(arguments.no_gate_dataset, (0, 1, 2)),
        "validation": NoGateDataset(arguments.no_gate_dataset, (3,)),
        "test": NoGateDataset(arguments.no_gate_dataset, (4,)),
    }
    if arguments.paired_no_gate_dataset:
        paired = {
            "train": range(16), "validation": range(16, 18), "test": range(18, 20)
        }
        negative = {
            split: ConcatDataset(
                (
                    dataset,
                    NoGateDataset(arguments.paired_no_gate_dataset, paired[split]),
                )
            )
            for split, dataset in negative.items()
        }
    real = {
        "train": RealGateDataset(arguments.real_root, ("flight_06",)),
        "validation": RealGateDataset(arguments.real_root, ("flight_07",)),
        "test": RealGateDataset(arguments.real_root, ("flight_08",)),
    }
    if arguments.limit > 0:
        for collection in (synthetic, negative, real):
            for split in collection:
                collection[split] = limited(collection[split], arguments.limit)
    return synthetic, negative, real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap", choices=tuple(TAP_CHANNELS), required=True)
    for name in (
        "checkpoint", "pulp-root", "gate-dataset", "gate-targets",
        "gate-split-file", "no-gate-dataset", "real-root", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--paired-no-gate-dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenDroNetV3GateHeads(
        arguments.checkpoint, arguments.pulp_root, arguments.tap
    ).to(device)
    synthetic, negative, real = make_datasets(arguments)

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=shuffle,
            num_workers=arguments.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=arguments.workers > 0,
        )

    training = {
        "synthetic": loader(synthetic["train"], True),
        "negative": loader(negative["train"], True),
        "real": loader(real["train"], True),
    }
    validation = (
        loader(synthetic["validation"], False),
        loader(negative["validation"], False),
        loader(real["validation"], False),
    )
    testing = (
        loader(synthetic["test"], False),
        loader(negative["test"], False),
        loader(real["test"], False),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )
    profile = profile_combined(model, device)
    print(
        json.dumps(
            {
                "device": str(device),
                "tap": arguments.tap,
                "profile": profile,
                "dataset_sizes": {
                    domain: {split: len(data) for split, data in collection.items()}
                    for domain, collection in (
                        ("synthetic", synthetic), ("negative", negative), ("real", real)
                    )
                },
            }
        ),
        flush=True,
    )

    best_score = float("inf")
    best_epoch = None
    history = []
    with (arguments.output / "log.csv").open("w", newline="") as stream:
        writer = None
        for epoch in range(1, arguments.epochs + 1):
            started = time.time()
            train_metrics = train_epoch(model, training, optimizer, device)
            validation_metrics = evaluate(model, *validation, device)
            model.assert_backbone_unchanged()
            score = (
                validation_metrics["synthetic_corner_mean_px"] / 10.0
                + (1.0 - validation_metrics["synthetic_gate_iou"])
                + validation_metrics["presence_bce"]
            )
            record = {
                "epoch": epoch,
                "seconds": time.time() - started,
                "selection_score": score,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            flat = {
                "epoch": epoch,
                "seconds": record["seconds"],
                "selection_score": score,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            stream.flush()
            print(json.dumps({"tap": arguments.tap, **record}), flush=True)
            state = {
                "architecture": "frozen_dronetv3_gate_heads",
                "tap": arguments.tap,
                "epoch": epoch,
                "model": model.state_dict(),
                "validation": validation_metrics,
                "selection_score": score,
                "backbone_checkpoint": str(arguments.checkpoint),
                "backbone_sha256": model.backbone_fingerprint,
                "frozen_backbone_verified": True,
                "temporal_input_policy": "repeat_current_frame",
                "profile": profile,
            }
            torch.save(state, arguments.output / "last.pt")
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(state, arguments.output / "best.pt")

    selected = torch.load(
        arguments.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(selected["model"], strict=True)
    final = {
        "architecture": "frozen_dronetv3_gate_heads",
        "tap": arguments.tap,
        "best_epoch": best_epoch,
        "selection": "minimum composite validation score; test untouched",
        "validation": selected["validation"],
        "test": evaluate(model, *testing, device),
        "profile": profile,
        "backbone_checkpoint": str(arguments.checkpoint),
        "backbone_sha256": model.assert_backbone_unchanged(),
        "frozen_backbone_verified": True,
        "temporal_input_policy": "repeat_current_frame",
        "real_split": {
            "train": ["flight_06"],
            "validation": ["flight_07"],
            "test": ["flight_08"],
        },
        "dataset_sizes": {
            domain: {split: len(data) for split, data in collection.items()}
            for domain, collection in (
                ("synthetic", synthetic), ("negative", negative), ("real", real)
            )
        },
        "history": history,
    }
    (arguments.output / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"final": final}), flush=True)


if __name__ == "__main__":
    main()
