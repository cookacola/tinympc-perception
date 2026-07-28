#!/usr/bin/env python3
"""Train resize-free DORY students from the rich STDC multi-head model."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCMultiTaskDataset
from gap8_perception.data_stdc_real import RealCornerDataset
from gap8_perception.losses import soft_dice_loss
from gap8_perception.losses_stdc import focal_heatmap_mse
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet
from gap8_perception.model_stdc_dory import (
    Gap8STDCCornerDoryNet,
    Gap8STDCDangerDoryNet,
    initialize_from_rich,
)
from gap8_perception.profile_stdc_dory import combined_profile
from gap8_perception.train_stdc import move


def conservative_danger_target(danger: torch.Tensor) -> torch.Tensor:
    return F.adaptive_max_pool2d(danger, (8, 10))


def run_epoch(
    corner,
    danger,
    teacher,
    loader,
    device,
    optimizer=None,
    real_loader=None,
    real_weight=2.0,
):
    training = optimizer is not None
    corner.train(training)
    danger.train(training)
    teacher.eval()
    totals = {"total": 0.0, "corner": 0.0, "danger": 0.0, "distill": 0.0}
    count = 0
    real_iterator = itertools.cycle(real_loader) if real_loader is not None else None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            corner_logits = corner(batch["image"])
            danger_logits = danger(batch["image"])
            danger_target = conservative_danger_target(batch["danger"])
            danger_binary = (danger_target >= 0.5).to(danger_target.dtype)
            corner_loss = focal_heatmap_mse(
                corner_logits, batch["corners"], batch["corner_valid"]
            )
            danger_loss = F.binary_cross_entropy_with_logits(
                danger_logits,
                danger_binary,
                pos_weight=danger_logits.new_tensor(2.5),
            ) + 0.5 * soft_dice_loss(danger_logits, danger_binary)
            with torch.no_grad():
                rich = teacher(batch["image"])
                rich_corner = rich["corners"].sigmoid()
                rich_danger = F.adaptive_max_pool2d(
                    rich["danger"].sigmoid(), (8, 10)
                )
            distill = F.smooth_l1_loss(
                corner_logits.sigmoid(), rich_corner
            ) + 2.0 * F.binary_cross_entropy_with_logits(
                danger_logits, rich_danger
            )
            real_corner_loss = corner_loss.new_zeros(())
            if real_iterator is not None:
                real_batch = move(next(real_iterator), device)
                real_corner_loss = focal_heatmap_mse(
                    corner(real_batch["image"]),
                    real_batch["corners"],
                    real_batch["corner_valid"],
                )
            total = (
                corner_loss
                + 2.0 * danger_loss
                + 0.25 * distill
                + real_weight * real_corner_loss
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(corner.parameters()) + list(danger.parameters()), 5.0
                )
                optimizer.step()
            size = batch["image"].shape[0]
            for key, value in (
                ("total", total),
                ("corner", corner_loss),
                ("danger", danger_loss),
                ("distill", distill),
            ):
                totals[key] += float(value.detach()) * size
            count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def real_corner_validation(corner, loader, device):
    corner.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            loss = focal_heatmap_mse(
                corner(batch["image"]),
                batch["corners"],
                batch["corner_valid"],
            )
            size = batch["image"].shape[0]
            total += float(loss) * size
            count += size
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--real-train-flights", default="flight_06")
    parser.add_argument("--real-validation-flights", default="flight_07")
    parser.add_argument("--real-weight", type=float, default=2.0)
    parser.add_argument(
        "--select-corner-last",
        action="store_true",
        help=(
            "Use the final corner epoch for a post-selection train+validation "
            "refit; danger remains selected on synthetic validation."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_state = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    teacher = Gap8STDCMultiHeadNet()
    teacher.load_state_dict(teacher_state["model"])
    corner, danger = Gap8STDCCornerDoryNet(), Gap8STDCDangerDoryNet()
    initialize_from_rich(corner, danger, teacher_state["model"])
    teacher, corner, danger = (
        teacher.to(device),
        corner.to(device),
        danger.to(device),
    )
    optimizer = torch.optim.AdamW(
        list(corner.parameters()) + list(danger.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    train = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "train", augment=True
    )
    validation = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "validation"
    )
    loaders = [
        DataLoader(
            dataset,
            args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for dataset, shuffle in ((train, True), (validation, False))
    ]
    real_loader = None
    real_validation_loader = None
    if args.real_root:
        real_dataset = RealCornerDataset(
            args.real_root,
            tuple(
                item for item in args.real_train_flights.split(",") if item
            ),
        )
        real_loader = DataLoader(
            real_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        real_validation_dataset = RealCornerDataset(
            args.real_root,
            tuple(
                item
                for item in args.real_validation_flights.split(",")
                if item
            ),
        )
        real_validation_loader = DataLoader(
            real_validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
    best = float("inf")
    best_corner = float("inf")
    best_danger = float("inf")
    log = (args.output / "log.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(
        ["epoch"]
        + [f"train_{key}" for key in ("total", "corner", "danger", "distill")]
        + [f"val_{key}" for key in ("total", "corner", "danger", "distill")]
        + ["real_val_corner", "selection_score"]
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            corner,
            danger,
            teacher,
            loaders[0],
            device,
            optimizer,
            real_loader=real_loader,
            real_weight=args.real_weight,
        )
        val_metrics = run_epoch(corner, danger, teacher, loaders[1], device)
        real_val_corner = (
            real_corner_validation(corner, real_validation_loader, device)
            if real_validation_loader is not None
            else val_metrics["corner"]
        )
        selection_score = real_val_corner + 0.10 * val_metrics["danger"]
        writer.writerow(
            [epoch]
            + [train_metrics[key] for key in ("total", "corner", "danger", "distill")]
            + [val_metrics[key] for key in ("total", "corner", "danger", "distill")]
            + [real_val_corner, selection_score]
        )
        log.flush()
        state = {
            "epoch": epoch,
            "corner_model": corner.state_dict(),
            "danger_model": danger.state_dict(),
            "optimizer": optimizer.state_dict(),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "real_corner_training": (
                {
                    "root": str(args.real_root),
                    "flights": [
                        item
                        for item in args.real_train_flights.split(",")
                        if item
                    ],
                    "validation_flights": [
                        item
                        for item in args.real_validation_flights.split(",")
                        if item
                    ],
                    "weight": args.real_weight,
                }
                if args.real_root
                else None
            ),
            "architecture": "Gap8STDCDoryPair",
            "corner_output": [4, 30, 40],
            "danger_output": [1, 8, 10],
            "danger_reduction": "adaptive_max_pool_from_15x20",
            "resource_profile": combined_profile(),
            "selection_score": selection_score,
            "best_selection_score": min(best, selection_score),
        }
        torch.save(state, args.output / "last.pt")
        if selection_score < best:
            best = selection_score
            state["best_selection_score"] = best
            torch.save(state, args.output / "best_total.pt")
        if real_val_corner < best_corner:
            best_corner = real_val_corner
            corner_state = dict(state)
            corner_state["best_real_validation_corner"] = best_corner
            torch.save(corner_state, args.output / "best_corner.pt")
        if val_metrics["danger"] < best_danger:
            best_danger = val_metrics["danger"]
            danger_state = dict(state)
            danger_state["best_synthetic_validation_danger"] = best_danger
            torch.save(danger_state, args.output / "best_danger.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "real_validation_corner": real_val_corner,
                    "selection_score": selection_score,
                }
            ),
            flush=True,
        )
    log.close()
    selected_corner_path = (
        args.output / "last.pt"
        if args.select_corner_last
        else args.output / "best_corner.pt"
    )
    selected_corner = torch.load(
        selected_corner_path, map_location="cpu", weights_only=False
    )
    selected_danger = torch.load(
        args.output / "best_danger.pt", map_location="cpu", weights_only=False
    )
    selected = dict(selected_corner)
    selected["corner_model"] = selected_corner["corner_model"]
    selected["danger_model"] = selected_danger["danger_model"]
    selected["selection"] = {
        "corner_checkpoint": str(selected_corner_path),
        "danger_checkpoint": str(args.output / "best_danger.pt"),
        "real_validation_corner": (
            selected_corner.get("best_real_validation_corner")
            if not args.select_corner_last
            else None
        ),
        "corner_selection": (
            "fixed_final_epoch_train_plus_validation_refit"
            if args.select_corner_last
            else "best_real_validation"
        ),
        "synthetic_validation_danger": selected_danger[
            "best_synthetic_validation_danger"
        ],
    }
    torch.save(selected, args.output / "selected.pt")


if __name__ == "__main__":
    main()
