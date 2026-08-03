#!/usr/bin/env python3
"""Train a stock-DORY-compatible shared encoder and two task heads."""

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
    Gap8STDCSharedDoryNet,
    initialize_shared_from_pair,
    initialize_shared_from_rich,
)
from gap8_perception.profile_stdc_dory import shared_profile
from gap8_perception.train_stdc import move


def run_epoch(
    model,
    teacher,
    loader,
    device,
    optimizer=None,
    real_loader=None,
    real_weight=2.0,
):
    training = optimizer is not None
    model.train(training)
    if getattr(model, "freeze_corner_path", False):
        model.stem.eval()
        model.stage1.eval()
        model.corner_head.eval()
    teacher.eval()
    real_iterator = itertools.cycle(real_loader) if real_loader else None
    totals = {"total": 0.0, "corner": 0.0, "danger": 0.0, "distill": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["image"])
            danger_target = F.adaptive_max_pool2d(batch["danger"], (8, 10))
            danger_binary = (danger_target >= 0.5).to(danger_target.dtype)
            corner_loss = focal_heatmap_mse(
                outputs["corners"], batch["corners"], batch["corner_valid"]
            )
            danger_loss = F.binary_cross_entropy_with_logits(
                outputs["danger"],
                danger_binary,
                pos_weight=outputs["danger"].new_tensor(2.5),
            ) + 0.5 * soft_dice_loss(outputs["danger"], danger_binary)
            with torch.no_grad():
                rich = teacher(batch["image"])
                rich_corner = rich["corners"].sigmoid()
                rich_danger = F.adaptive_max_pool2d(
                    rich["danger"].sigmoid(), (8, 10)
                )
            distill = F.smooth_l1_loss(
                outputs["corners"].sigmoid(), rich_corner
            ) + 2.0 * F.binary_cross_entropy_with_logits(
                outputs["danger"], rich_danger
            )
            real_corner_loss = corner_loss.new_zeros(())
            if real_iterator is not None:
                real = move(next(real_iterator), device)
                real_corner_loss = focal_heatmap_mse(
                    model(real["image"])["corners"],
                    real["corners"],
                    real["corner_valid"],
                )
            total = (
                corner_loss
                + 2.0 * danger_loss
                + 0.25 * distill
                + real_weight * real_corner_loss
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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


def real_corner_loss(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            loss = focal_heatmap_mse(
                model(batch["image"])["corners"],
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
    parser.add_argument("--initial-pair-checkpoint", type=Path)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--real-weight", type=float, default=2.0)
    parser.add_argument("--freeze-corner-path", action="store_true")
    parser.add_argument("--real-train-flights", default="flight_06")
    parser.add_argument("--real-validation-flights", default="flight_07")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_state = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    teacher = Gap8STDCMultiHeadNet()
    teacher.load_state_dict(teacher_state["model"])
    model = Gap8STDCSharedDoryNet()
    initialize_shared_from_rich(model, teacher_state["model"])
    if args.initial_pair_checkpoint:
        pair_state = torch.load(
            args.initial_pair_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        initialize_shared_from_pair(
            model, pair_state["corner_model"], pair_state["danger_model"]
        )
    model.freeze_corner_path = args.freeze_corner_path
    if args.freeze_corner_path:
        for module in (model.stem, model.stage1, model.corner_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    teacher, model = teacher.to(device), model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )
    synthetic_train = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "train", augment=True
    )
    synthetic_validation = STDCMultiTaskDataset(
        args.dataset, args.targets, args.split_file, "validation"
    )
    real_train = RealCornerDataset(
        args.real_root,
        tuple(item for item in args.real_train_flights.split(",") if item),
    )
    real_validation = RealCornerDataset(
        args.real_root,
        tuple(
            item for item in args.real_validation_flights.split(",") if item
        ),
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
        for dataset, shuffle in (
            (synthetic_train, True),
            (synthetic_validation, False),
            (real_train, True),
            (real_validation, False),
        )
    ]
    best = float("inf")
    with (args.output / "log.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(
            ["epoch"]
            + [f"train_{key}" for key in ("total", "corner", "danger", "distill")]
            + [f"val_{key}" for key in ("total", "corner", "danger", "distill")]
            + ["real_val_corner", "selection_score"]
        )
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(
                model,
                teacher,
                loaders[0],
                device,
                optimizer,
                None if args.freeze_corner_path else loaders[2],
                args.real_weight,
            )
            validation = run_epoch(model, teacher, loaders[1], device)
            real_validation_corner = real_corner_loss(
                model, loaders[3], device
            )
            selection_score = (
                real_validation_corner + 0.10 * validation["danger"]
            )
            writer.writerow(
                [epoch]
                + [train[key] for key in ("total", "corner", "danger", "distill")]
                + [
                    validation[key]
                    for key in ("total", "corner", "danger", "distill")
                ]
                + [real_validation_corner, selection_score]
            )
            log.flush()
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "architecture": "Gap8STDCSharedDoryNet",
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "initial_pair_checkpoint": (
                    str(args.initial_pair_checkpoint)
                    if args.initial_pair_checkpoint
                    else None
                ),
                "real_train_flights": args.real_train_flights.split(","),
                "real_validation_flights": (
                    args.real_validation_flights.split(",")
                ),
                "held_out_real_flight": "flight_08",
                "freeze_corner_path": args.freeze_corner_path,
                "resource_profile": shared_profile(),
                "selection_score": selection_score,
                "best_selection_score": min(best, selection_score),
            }
            torch.save(state, args.output / "last.pt")
            if selection_score < best:
                best = selection_score
                state["best_selection_score"] = best
                torch.save(state, args.output / "best_total.pt")
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train,
                        "validation": validation,
                        "real_validation_corner": real_validation_corner,
                        "selection_score": selection_score,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
