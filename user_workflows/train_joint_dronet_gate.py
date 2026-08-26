#!/usr/bin/env python3
"""Mixed navigation/gate fine-tuning of the best DroNetV3 gate tap."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_frozen_dronet_gate_heads import (
    FrozenDroNetV3GateHeads,
    average_precision,
    binary_auroc,
    evaluate as evaluate_gate,
    gate_loss,
    infinite,
    make_datasets,
    profile_combined,
)


@dataclass(frozen=True)
class PairSample:
    previous: Path
    current: Path
    yaw_rate: float
    collision: float


def find_strict_pairs(root: Path, partition: str):
    """Match the fair benchmark's adjacent-rank, same-acquisition rule."""
    samples = []
    for labels_path in sorted(root.rglob("labels_partitioned.csv")):
        with labels_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {
            "filename", "label_yaw_rate", "label_collision", "partition"
        }.issubset(rows[0]):
            continue
        images = labels_path.parent / "images"
        disk_files = sorted(
            (path for path in images.glob("*.jpeg") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        rank = {path.name: index for index, path in enumerate(disk_files)}
        for previous, current in zip(rows, rows[1:]):
            if previous["partition"] != partition or current["partition"] != partition:
                continue
            previous_name = previous["filename"]
            current_name = current["filename"]
            if previous_name not in rank or current_name not in rank:
                continue
            if rank[current_name] != rank[previous_name] + 1:
                continue
            samples.append(
                PairSample(
                    previous=images / previous_name,
                    current=images / current_name,
                    yaw_rate=float(current["label_yaw_rate"]) / 90.0,
                    collision=float(current["label_collision"]),
                )
            )
    return samples


class MatchedFrames(Dataset):
    def __init__(self, samples, augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            for path in (sample.previous, sample.current)
        ]
        if any(image is None for image in images):
            raise FileNotFoundError(sample.current)
        height, width = images[0].shape
        if width < 200 or height < 200:
            raise ValueError(f"image smaller than 200x200: {sample.current}")
        yaw = sample.yaw_rate
        if self.augment:
            top = random.randint(0, height - 200)
            left = random.randint(0, width - 200)
            brightness = random.uniform(0.8, 1.2)
            flip = random.random() < 0.5
        else:
            top = (height - 200) // 2
            left = (width - 200) // 2
            brightness = 1.0
            flip = False
        frames = []
        for image in images:
            image = image[top : top + 200, left : left + 200]
            image = np.clip(image.astype(np.float32) * brightness, 0, 255)
            if flip:
                image = image[:, ::-1]
            frames.append(torch.from_numpy(image.copy()).float() / 255.0)
        if flip:
            yaw = -yaw
        return (
            torch.stack(frames),
            torch.tensor((yaw, sample.collision), dtype=torch.float32),
        )


def navigation_metrics(model, loader, device):
    model.eval()
    yaw_prediction = []
    yaw_truth = []
    collision_score = []
    collision_truth = []
    total_bce = 0.0
    examples = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model.forward_navigation(frames)
            total_bce += F.binary_cross_entropy(
                output[1], labels[:, 1], reduction="sum"
            ).item()
            yaw_prediction.append(output[0].cpu().numpy())
            yaw_truth.append(labels[:, 0].cpu().numpy())
            collision_score.append(output[1].cpu().numpy())
            collision_truth.append(labels[:, 1].cpu().numpy())
            examples += len(labels)
    yaw_prediction = np.concatenate(yaw_prediction).astype(np.float64)
    yaw_truth = np.concatenate(yaw_truth).astype(np.float64)
    score = np.concatenate(collision_score).astype(np.float64)
    truth = np.concatenate(collision_truth).astype(bool)
    prediction = score >= 0.5
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "examples": examples,
        "yaw_rmse": float(np.sqrt(np.mean(np.square(yaw_prediction - yaw_truth)))),
        "collision_bce": total_bce / examples,
        "collision_auroc": binary_auroc(truth, score),
        "collision_ap": average_precision(truth, score),
        "collision_accuracy_at_0_5": float((prediction == truth).mean()),
        "collision_f1_at_0_5": 2.0 * precision * recall
        / max(1e-12, precision + recall),
        "collision_precision_at_0_5": precision,
        "collision_recall_at_0_5": recall,
    }


def navigation_loss(model, batch, device):
    frames, labels = batch
    frames = frames.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    output = model.forward_navigation(frames)
    yaw = F.mse_loss(output[0], labels[:, 0])
    collision = F.binary_cross_entropy(output[1], labels[:, 1])
    return yaw + collision, {"navigation_yaw_mse": yaw, "navigation_bce": collision}


def train_epoch(model, loaders, optimizer, device, navigation_weight):
    model.train()
    streams = {name: infinite(loader) for name, loader in loaders.items()}
    steps = max(len(loader) for loader in loaders.values())
    totals = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        navigation, navigation_parts = navigation_loss(
            model, next(streams["navigation"]), device
        )
        gate, gate_parts = gate_loss(
            model,
            next(streams["synthetic"]),
            next(streams["real"]),
            next(streams["negative"]),
            device,
        )
        loss = navigation_weight * navigation + gate
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite mixed loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        values = {
            "total": loss,
            "navigation": navigation,
            "gate": gate,
            **navigation_parts,
            **{f"gate_{name}": value for name, value in gate_parts.items()},
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / steps for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint", "pulp-root", "navigation-data", "baseline-result",
        "gate-dataset", "gate-targets", "gate-split-file", "no-gate-dataset",
        "real-root", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--paired-no-gate-dataset", type=Path)
    parser.add_argument("--tap", choices=("block1", "block2", "block3"), default="block2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--navigation-batch-size", type=int, default=256)
    parser.add_argument("--gate-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--navigation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gate-learning-rate", type=float, default=3e-4)
    parser.add_argument("--navigation-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    navigation_samples = {
        split: find_strict_pairs(arguments.navigation_data, split)
        for split in ("train", "valid", "test")
    }
    if arguments.limit > 0:
        navigation_samples = {
            split: values[: arguments.limit]
            for split, values in navigation_samples.items()
        }
    synthetic, negative, real = make_datasets(arguments)

    model = FrozenDroNetV3GateHeads(
        Path(torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)["backbone_checkpoint"]),
        arguments.pulp_root,
        arguments.tap,
    )
    initial = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(initial["model"], strict=True)
    model.set_backbone_trainable().to(device)

    def navigation_loader(split, shuffle):
        return DataLoader(
            MatchedFrames(navigation_samples[split], augment=shuffle),
            batch_size=arguments.navigation_batch_size,
            shuffle=shuffle,
            num_workers=arguments.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=arguments.workers > 0,
        )

    def gate_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=arguments.gate_batch_size,
            shuffle=shuffle,
            num_workers=arguments.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=arguments.workers > 0,
        )

    training = {
        "navigation": navigation_loader("train", True),
        "synthetic": gate_loader(synthetic["train"], True),
        "negative": gate_loader(negative["train"], True),
        "real": gate_loader(real["train"], True),
    }
    navigation_validation = navigation_loader("valid", False)
    navigation_test = navigation_loader("test", False)
    gate_validation = (
        gate_loader(synthetic["validation"], False),
        gate_loader(negative["validation"], False),
        gate_loader(real["validation"], False),
    )
    gate_test = (
        gate_loader(synthetic["test"], False),
        gate_loader(negative["test"], False),
        gate_loader(real["test"], False),
    )

    baseline = json.loads(arguments.baseline_result.read_text())
    baseline_navigation_validation = baseline["validation"]
    baseline_navigation_test = baseline["test"]
    baseline_gate_validation = evaluate_gate(model, *gate_validation, device)
    baseline_gate_test = initial["validation"] if arguments.limit > 0 else None
    profile = profile_combined(model, device)

    fc_ids = {id(parameter) for parameter in model.backbone.fc.parameters()}
    encoder_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if id(parameter) not in fc_ids
    ]
    gate_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": arguments.encoder_learning_rate},
            {"params": model.backbone.fc.parameters(), "lr": arguments.navigation_learning_rate},
            {"params": gate_parameters, "lr": arguments.gate_learning_rate},
        ],
        weight_decay=1e-4,
    )

    best_key = None
    history = []
    with (arguments.output / "log.csv").open("w", newline="") as stream:
        writer = None
        for epoch in range(1, arguments.epochs + 1):
            started = time.time()
            train = train_epoch(
                model, training, optimizer, device, arguments.navigation_weight
            )
            navigation = navigation_metrics(model, navigation_validation, device)
            gate = evaluate_gate(model, *gate_validation, device)
            violations = {
                "yaw_rmse": max(
                    0.0,
                    navigation["yaw_rmse"]
                    - 1.02 * baseline_navigation_validation["yaw_rmse"],
                ),
                "collision_auroc": max(
                    0.0,
                    baseline_navigation_validation["collision_auroc"]
                    - 0.01
                    - navigation["collision_auroc"],
                ),
                "collision_ap": max(
                    0.0,
                    baseline_navigation_validation["collision_ap"]
                    - 0.01
                    - navigation["collision_ap"],
                ),
            }
            feasible = not any(violations.values())
            gate_score = (
                gate["synthetic_corner_mean_px"] / 10.0
                + (1.0 - gate["synthetic_gate_iou"])
                + gate["presence_bce"]
            )
            violation_score = sum(violations.values())
            selection_key = (
                0 if feasible else 1,
                gate_score if feasible else violation_score,
                gate_score,
            )
            record = {
                "epoch": epoch,
                "seconds": time.time() - started,
                "train": train,
                "navigation_validation": navigation,
                "gate_validation": gate,
                "selection_feasible": feasible,
                "constraint_violations": violations,
                "gate_selection_score": gate_score,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            flat = {
                "epoch": epoch,
                "seconds": record["seconds"],
                "selection_feasible": int(feasible),
                "gate_selection_score": gate_score,
                **{f"train_{name}": value for name, value in train.items()},
                **{f"nav_{name}": value for name, value in navigation.items()},
                **{f"gate_{name}": value for name, value in gate.items()},
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            stream.flush()
            state = {
                "architecture": "joint_dronetv3_gate_heads",
                "tap": arguments.tap,
                "epoch": epoch,
                "model": model.state_dict(),
                "record": record,
                "profile": profile,
                "source_frozen_checkpoint": str(arguments.checkpoint),
            }
            torch.save(state, arguments.output / "last.pt")
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                torch.save(state, arguments.output / "best.pt")

    selected = torch.load(
        arguments.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(selected["model"], strict=True)
    final = {
        "architecture": "joint_dronetv3_gate_heads",
        "tap": arguments.tap,
        "best_epoch": selected["epoch"],
        "selection": "gate composite subject to navigation validation constraints; test untouched",
        "baseline_navigation_validation": baseline_navigation_validation,
        "baseline_navigation_test": baseline_navigation_test,
        "baseline_frozen_gate_validation": baseline_gate_validation,
        "baseline_frozen_gate_test": baseline_gate_test,
        "joint_navigation_validation": navigation_metrics(
            model, navigation_validation, device
        ),
        "joint_navigation_test": navigation_metrics(model, navigation_test, device),
        "joint_gate_validation": evaluate_gate(model, *gate_validation, device),
        "joint_gate_test": evaluate_gate(model, *gate_test, device),
        "profile": profile,
        "navigation_counts": {
            split: len(values) for split, values in navigation_samples.items()
        },
        "real_split": {
            "train": ["flight_06"],
            "validation": ["flight_07"],
            "test": ["flight_08"],
        },
        "optimization": {
            "encoder_learning_rate": arguments.encoder_learning_rate,
            "navigation_learning_rate": arguments.navigation_learning_rate,
            "gate_learning_rate": arguments.gate_learning_rate,
            "navigation_weight": arguments.navigation_weight,
        },
        "history": history,
    }
    (arguments.output / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"final": final}), flush=True)


if __name__ == "__main__":
    main()
