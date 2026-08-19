#!/usr/bin/env python3
"""Distill the selected two-frame ESPNet into a stock-DORY student."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

TRAINING_REPO = Path("/home/cchen/tinympc-gate-texture")
ISAAC_REPO = Path("/home/cchen/isaacsim-workspace")
sys.path[:0] = [str(ISAAC_REPO), str(TRAINING_REPO)]

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

import gap8_perception as _gap8_package
_gap8_package.__path__.append(str(TRAINING_REPO / "gap8_perception"))
from gap8_perception.losses import soft_dice_loss, weighted_corner_mse
from gap8_perception.model_espnet_dory_student import ESPNetDoryStudent

from gap8_perception.data import MultiTaskDataset  # noqa: E402
from gap8_perception.evaluate import local_centroid  # noqa: E402
from gap8_perception.temporal_data import TemporalHorizonDataset  # noqa: E402
from user_workflows.train_retained_obstacle_gate import (  # noqa: E402
    NoGateDataset,
    RealGateDataset,
    RetainedObstacleGateModel,
)


def infinite(loader):
    while True:
        yield from loader


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    ordered = labels[np.argsort(-np.asarray(scores))]
    return float(
        (np.cumsum(ordered) / np.arange(1, len(ordered) + 1))[ordered].mean()
    )


def danger_target(batch, teacher, device):
    inverse_depth = batch["inverse_depth"].to(device).clamp(0.0, 1.0)
    valid = batch["depth_valid"].to(device).to(inverse_depth.dtype)
    near = F.adaptive_max_pool2d(inverse_depth * valid, (10, 10))
    with torch.no_grad():
        corridor = teacher.obstacle.projector(
            batch["horizon"].to(device),
            batch["horizon_mask"].to(device),
            10,
            10,
        ).amax(1, keepdim=True)
    return near, corridor


def gate_loss(model, synthetic, negative, real, device):
    image = synthetic["image"].to(device)
    frames = image.repeat(1, 2, 1, 1)
    output = model(frames)
    target = synthetic["gate"].to(device)
    valid = synthetic["corner_valid"].to(device)
    corner = weighted_corner_mse(
        output["corners"], synthetic["corners"].to(device), valid
    )
    mask = F.binary_cross_entropy_with_logits(output["gate"], target)
    mask = mask + 0.5 * soft_dice_loss(output["gate"], target)
    negative_image = negative["image"].to(device).repeat(1, 2, 1, 1)
    negative_mask = F.binary_cross_entropy_with_logits(
        model(negative_image)["gate"],
        torch.zeros((len(negative_image), 1, 40, 40), device=device),
    )
    real_image = real["image"].to(device).repeat(1, 2, 1, 1)
    real_output = model(real_image)
    real_corner = weighted_corner_mse(
        real_output["corners"],
        real["corners"].to(device),
        real["corner_valid"].to(device),
    )
    total = 10.0 * corner + mask + negative_mask + 5.0 * real_corner
    return total, {
        "corner": corner,
        "mask": mask,
        "negative_mask": negative_mask,
        "real_corner": real_corner,
    }


def train_epoch(model, teacher, loaders, optimizer, device):
    model.train()
    streams = {name: infinite(loader) for name, loader in loaders.items()}
    steps = max(len(loaders["obstacle"]), len(loaders["synthetic"]))
    totals = {}
    for _ in range(steps):
        obstacle = next(streams["obstacle"])
        frames = obstacle["images"].to(device)
        output = model(frames)
        near, corridor = danger_target(obstacle, teacher, device)
        map_loss = F.binary_cross_entropy_with_logits(output["danger"], near)
        corridor_score = (
            output["danger"].sigmoid() * corridor
        ).flatten(1).amax(1)
        collision_loss = F.binary_cross_entropy(
            corridor_score.clamp(1e-6, 1 - 1e-6),
            obstacle["collision"].to(device).float(),
        )
        auxiliary, parts = gate_loss(
            model,
            next(streams["synthetic"]),
            next(streams["negative"]),
            next(streams["real"]),
            device,
        )
        loss = 3.0 * map_loss + 2.0 * collision_loss + auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        values = {
            "total": loss,
            "danger_map": map_loss,
            "collision": collision_loss,
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def obstacle_raw_predictions(model, teacher, loader, device):
    model.eval()
    labels, scores = [], []
    intersection = union = 0
    for batch in loader:
        output = model(batch["images"].to(device))
        near, corridor = danger_target(batch, teacher, device)
        prediction = output["danger"].sigmoid()
        target_binary = near >= 0.5
        predicted_binary = prediction >= 0.5
        intersection += int((predicted_binary & target_binary).sum())
        union += int((predicted_binary | target_binary).sum())
        frame_score = (prediction * corridor).flatten(1).amax(1)
        labels.extend(batch["collision"].bool().tolist())
        scores.extend(frame_score.cpu().tolist())
    return np.asarray(labels, bool), np.asarray(scores), intersection, union


def obstacle_report(raw, threshold):
    labels_np, scores_np, intersection, union = raw
    prediction = scores_np >= threshold
    recall = float(prediction[labels_np].mean()) if labels_np.any() else float("nan")
    return {
        "danger_iou": intersection / max(1, union),
        "collision_auroc": auc(labels_np, scores_np),
        "collision_ap": average_precision(labels_np, scores_np),
        "collision_recall": recall,
        "collision_threshold": float(threshold),
        "collision_positive_count": int(labels_np.sum()),
        "examples": len(labels_np),
    }


def select_collision_threshold(model, teacher, loader, device, minimum_recall=0.95):
    raw = obstacle_raw_predictions(model, teacher, loader, device)
    candidates = np.linspace(0.01, 0.99, 99)
    reports = [
        obstacle_report(raw, float(threshold))
        for threshold in candidates
    ]
    feasible = [report for report in reports if report["collision_recall"] >= minimum_recall]
    return max(feasible or reports, key=lambda report: report["collision_threshold"])[
        "collision_threshold"
    ]


def evaluate_obstacle(model, teacher, loader, device, threshold=0.5):
    report = obstacle_report(
        obstacle_raw_predictions(model, teacher, loader, device), threshold
    )
    # Retain the original fixed-threshold field for checkpoint-selection compatibility.
    if threshold == 0.5:
        report["collision_recall_at_0_5"] = report["collision_recall"]
    return report


@torch.no_grad()
def evaluate_gate(model, synthetic_loader, negative_loader, real_loader, device):
    model.eval()
    intersection = union = negative_pixels = negative_total = 0
    errors, real_errors = [], []
    for batch in synthetic_loader:
        output = model(batch["image"].to(device).repeat(1, 2, 1, 1))
        target = batch["gate"].to(device) >= 0.5
        prediction = output["gate"].sigmoid() >= 0.5
        intersection += int((prediction & target).sum())
        union += int((prediction | target).sum())
        valid = batch["corner_valid"].to(device)
        points = local_centroid(output["corners"].sigmoid()) * 4
        if valid.any():
            errors.append(torch.linalg.vector_norm(
                points[valid] - batch["corner_xy"].to(device)[valid], dim=2
            ).cpu())
    for batch in negative_loader:
        output = model(batch["image"].to(device).repeat(1, 2, 1, 1))
        prediction = output["gate"].sigmoid() >= 0.5
        negative_pixels += int(prediction.sum())
        negative_total += prediction.numel()
    for batch in real_loader:
        output = model(batch["image"].to(device).repeat(1, 2, 1, 1))
        points = local_centroid(output["corners"].sigmoid()) * 4
        real_errors.append(torch.linalg.vector_norm(
            points - batch["corner_xy"].to(device), dim=2
        ).cpu())
    error = torch.cat(errors)
    real_error = torch.cat(real_errors)
    return {
        "gate_iou": intersection / max(1, union),
        "corner_mean_px": float(error.mean()),
        "corner_p95_px": float(torch.quantile(error, 0.95)),
        "real_corner_mean_px": float(real_error.mean()),
        "real_corner_p95_px": float(torch.quantile(real_error, 0.95)),
        "negative_mask_pixel_rate": negative_pixels / max(1, negative_total),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "obstacle-dataset", "gate-dataset", "gate-targets", "gate-split-file",
        "no-gate-dataset", "paired-no-gate-dataset", "real-root",
        "obstacle-checkpoint", "gate-checkpoint", "teacher-checkpoint", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = json.loads(
        (args.obstacle_dataset / "dataset_manifest.json").read_text()
    )["camera_calibration"]
    teacher = RetainedObstacleGateModel(
        args.obstacle_checkpoint, args.gate_checkpoint, camera
    ).to(device)
    teacher.load_state_dict(torch.load(
        args.teacher_checkpoint, map_location=device, weights_only=False
    )["model"])
    teacher.eval().requires_grad_(False)
    model = ESPNetDoryStudent().to(device)

    obstacle = {
        split: TemporalHorizonDataset(
            args.obstacle_dataset, split, 2, augment=split == "train",
            minimum_current_index=2,
        ) for split in ("train", "validation", "test")
    }
    synthetic = {
        split: MultiTaskDataset(
            args.gate_dataset, args.gate_targets, args.gate_split_file, split
        ) for split in ("train", "validation", "test")
    }
    base_negative = {
        "train": NoGateDataset(args.no_gate_dataset, (0, 1, 2)),
        "validation": NoGateDataset(args.no_gate_dataset, (3,)),
        "test": NoGateDataset(args.no_gate_dataset, (4,)),
    }
    paired_indices = {
        "train": range(16), "validation": range(16, 18), "test": range(18, 20)
    }
    negative = {
        split: ConcatDataset((
            base_negative[split],
            NoGateDataset(args.paired_no_gate_dataset, paired_indices[split]),
        )) for split in base_negative
    }
    real = {
        "train": RealGateDataset(args.real_root, ("flight_06",)),
        "validation": RealGateDataset(args.real_root, ("flight_07",)),
        "test": RealGateDataset(args.real_root, ("flight_08",)),
    }

    def loader(dataset, shuffle=False):
        return DataLoader(
            dataset, args.batch_size, shuffle=shuffle, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )

    train_loaders = {
        "obstacle": loader(obstacle["train"], True),
        "synthetic": loader(synthetic["train"], True),
        "negative": loader(negative["train"], True),
        "real": loader(real["train"], True),
    }
    validation = (
        loader(obstacle["validation"]), loader(synthetic["validation"]),
        loader(negative["validation"]), loader(real["validation"]),
    )
    test = (
        loader(obstacle["test"]), loader(synthetic["test"]),
        loader(negative["test"]), loader(real["test"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_key = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, teacher, train_loaders, optimizer, device)
        obstacle_validation = evaluate_obstacle(model, teacher, validation[0], device)
        gate_validation = evaluate_gate(model, *validation[1:], device)
        feasible = obstacle_validation["collision_recall_at_0_5"] >= 0.95
        score = (
            gate_validation["real_corner_mean_px"]
            + 10.0 * (1.0 - gate_validation["gate_iou"])
            + 10.0 * (1.0 - obstacle_validation["danger_iou"])
        )
        key = (0 if feasible else 1, score)
        record = {
            "epoch": epoch, "train": train,
            "obstacle_validation": obstacle_validation,
            "gate_validation": gate_validation,
            "selection_feasible": feasible, "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {
            "epoch": epoch, "model": model.state_dict(), "record": record,
            "architecture": "ESPNetDoryStudent", "teacher": str(args.teacher_checkpoint),
        }
        torch.save(state, args.output / "last.pt")
        torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
        if best_key is None or key < best_key:
            best_key = key
            torch.save(state, args.output / "best.pt")

    selected = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    selected_danger_threshold = select_collision_threshold(
        model, teacher, validation[0], device
    )
    summary = {
        "selected_epoch": selected["epoch"],
        "architecture": "ESPNetDoryStudent",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "teacher": str(args.teacher_checkpoint),
        "history": history,
        "selected_danger_threshold": selected_danger_threshold,
        "obstacle_validation_at_selected_threshold": evaluate_obstacle(
            model, teacher, validation[0], device, selected_danger_threshold
        ),
        "obstacle_test": evaluate_obstacle(
            model, teacher, test[0], device, selected_danger_threshold
        ),
        "gate_test": evaluate_gate(model, *test[1:], device),
        "real_split": {"train": "flight_06", "validation": "flight_07", "test": "flight_08"},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
