#!/usr/bin/env python3
"""Distill the privileged inverse-depth teacher into the mono STDC student."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gap8_perception.data_stdc import STDCPrivilegedDataset
from gap8_perception.losses_stdc import design_multitask_loss
from gap8_perception.model_stdc import (
    Gap8STDCMultiHeadNet,
    Gap8STDCPrivilegedTeacher,
)
from gap8_perception.train_stdc import move


def run_epoch(student, teacher, loader, device, optimizer=None, alpha=0.35):
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    totals = {"total": 0.0, "supervised": 0.0, "distill": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            student_outputs = student(batch["image"])
            with torch.no_grad():
                teacher_outputs = teacher(batch["privileged_image"])
            supervised = design_multitask_loss(student_outputs, batch)["total"]
            corner_distill = F.smooth_l1_loss(
                student_outputs["corners"].sigmoid(),
                teacher_outputs["corners"].sigmoid(),
            )
            danger_distill = F.binary_cross_entropy_with_logits(
                student_outputs["danger"],
                teacher_outputs["danger"].sigmoid(),
            )
            distill = corner_distill + 2.0 * danger_distill
            total = (1.0 - alpha) * supervised + alpha * distill
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                optimizer.step()
            size = batch["image"].shape[0]
            for key, value in (
                ("total", total),
                ("supervised", supervised),
                ("distill", distill),
            ):
                totals[key] += float(value.detach()) * size
            count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_state = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False
    )
    teacher_state = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    student = Gap8STDCMultiHeadNet()
    student.load_state_dict(student_state["model"])
    teacher = Gap8STDCPrivilegedTeacher()
    teacher.load_state_dict(teacher_state["model"])
    student, teacher = student.to(device), teacher.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
    train = STDCPrivilegedDataset(
        args.dataset, args.targets, args.split_file, "train", augment=False
    )
    validation = STDCPrivilegedDataset(
        args.dataset, args.targets, args.split_file, "validation", augment=False
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
    best = float("inf")
    log = (args.output / "log.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(
        [
            "epoch",
            "train_total",
            "train_supervised",
            "train_distill",
            "val_total",
            "val_supervised",
            "val_distill",
        ]
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            student, teacher, loaders[0], device, optimizer, args.alpha
        )
        val_metrics = run_epoch(
            student, teacher, loaders[1], device, alpha=args.alpha
        )
        writer.writerow(
            [epoch]
            + [train_metrics[key] for key in ("total", "supervised", "distill")]
            + [val_metrics[key] for key in ("total", "supervised", "distill")]
        )
        log.flush()
        state = {
            "epoch": epoch,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": "Gap8STDCMultiHeadNet",
            "distilled_from": str(args.teacher_checkpoint),
            "distillation_alpha": args.alpha,
            "input": [1, 120, 160],
            "best_total": min(best, val_metrics["total"]),
        }
        torch.save(state, args.output / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            state["best_total"] = best
            torch.save(state, args.output / "best_total.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": val_metrics,
                }
            ),
            flush=True,
        )
    log.close()


if __name__ == "__main__":
    main()
