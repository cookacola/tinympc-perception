#!/usr/bin/env python3
"""Refine only the DORY motion-TTC head on the natural corpus distribution."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train_ttc_gate_head import atomic_torch_save, finite_json, seed_everything
from .train_ttc_gate_joint_finetune import (
    TEST_LIMITS,
    VALIDATION_LIMITS,
    evaluate,
    move_batch,
)
from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_dory_model import DoryPartitionedMotionGateTTCNet
from .ttc_motion_gate_model import MotionConditionedESPNetInverseTTCNet
from .ttc_motion_losses import (
    critical_motion_conditioned_ttc_loss,
    parent_distillation_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=2.0)
    parser.add_argument("--ttc-refinements", type=int, default=7)
    parser.add_argument("--critical-regression-weight", type=float, default=1.0)
    parser.add_argument("--critical-risk-weight", type=float, default=0.5)
    parser.add_argument("--critical-positive-weight", type=float, default=4.0)
    parser.add_argument("--general-guard-multiplier", type=float, default=1.15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def train_epoch(model, teacher, loader, optimizer, device, args):
    model.eval()
    model.ttc_head.train()
    total, supervised_total, distill_total, examples = 0.0, 0.0, 0.0, 0
    for raw in loader:
        target = move_batch(raw, device)
        with torch.no_grad():
            teacher_output = teacher(target["images"], target["onboard_state"])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(target["images"], target["onboard_state"])
        supervised, _parts = critical_motion_conditioned_ttc_loss(
            prediction,
            target,
            critical_regression_weight=args.critical_regression_weight,
            critical_risk_weight=args.critical_risk_weight,
            critical_positive_weight=args.critical_positive_weight,
        )
        distillation, _distill_parts = parent_distillation_loss(prediction, teacher_output)
        loss = supervised + args.distillation_weight * distillation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.ttc_head.parameters(), 5.0)
        optimizer.step()
        count = target["images"].shape[0]
        examples += count
        total += float(loss.detach()) * count
        supervised_total += float(supervised.detach()) * count
        distill_total += float(distillation.detach()) * count
    return {
        "loss": total / examples,
        "supervised_ttc": supervised_total / examples,
        "distillation": distill_total / examples,
    }


def selection_key(validation, general_guard_multiplier=1.15):
    values = validation["ttc"]
    normalized_error = sum(
        values[name] / VALIDATION_LIMITS[name]
        for name in (
            "inverse_ttc_mae_s_inv", "approaching_inverse_ttc_mae_s_inv",
            "inverse_depth_mae_m_inv", "flow_epe_cells",
        )
    )
    regression_precision = values["regression_critical_precision"]
    regression_recall = values["regression_critical_recall"]
    regression_f1 = (
        2 * regression_precision * regression_recall
        / max(regression_precision + regression_recall, 1e-12)
    )
    general_guard_passed = all(
        values[name] <= general_guard_multiplier * VALIDATION_LIMITS[name]
        for name in (
            "inverse_ttc_mae_s_inv", "approaching_inverse_ttc_mae_s_inv",
            "inverse_depth_mae_m_inv", "flow_epe_cells",
        )
    )
    return (
        int(validation["retention_passed"]),
        int(general_guard_passed),
        -values["critical_inverse_ttc_mae_s_inv"],
        regression_f1,
        -values["approaching_inverse_ttc_mae_s_inv"],
        -normalized_error,
        values["critical_recall_at_0_552"],
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DORY TTC refinement requires a Slurm GPU allocation")
    train_set = TTCGateDataset(args.dataset, "train", augment=True)
    validation_set = TTCGateDataset(args.dataset, "validation")
    test_set = TTCGateDataset(args.dataset, "test")
    options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **options
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)

    saved = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    model = DoryPartitionedMotionGateTTCNet(
        ttc_refinements=args.ttc_refinements
    ).to(device)
    expansion = model.initialize_from_shallower_checkpoint(args.initial_checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.ttc_head.parameters():
        parameter.requires_grad_(True)
    teacher = MotionConditionedESPNetInverseTTCNet().to(device)
    teacher_initialization = teacher.initialize_from_checkpoint(args.teacher_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.ttc_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    baseline = evaluate(model, validation_loader, device)
    summary = {
        "experiment": "dory_motion_ttc_natural_refinement_v1",
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_epoch": saved.get("epoch"),
        "ttc_head_expansion": expansion,
        "teacher_initialization": teacher_initialization,
        "teacher_is_training_only": True,
        "trainable": ["ttc_head"],
        "encoder_and_gate_frozen": True,
        "natural_sampling": True,
        "distillation_weight": args.distillation_weight,
        "critical_objective": {
            "critical_regression_weight": args.critical_regression_weight,
            "critical_risk_weight": args.critical_risk_weight,
            "critical_positive_weight": args.critical_positive_weight,
            "general_guard_multiplier": args.general_guard_multiplier,
            "selection_priority": [
                "complete_retention", "critical_inverse_ttc_mae",
                "regression_critical_f1", "approaching_inverse_ttc_mae",
                "aggregate_parent_error", "risk_head_critical_recall",
            ],
        },
        "baseline_validation": baseline,
        "validation_limits": VALIDATION_LIMITS,
        "test_limits": TEST_LIMITS,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)
    best_path = args.output_dir / "dory_motion_gate_ttc_refined_best.pt"
    latest_path = args.output_dir / "dory_motion_gate_ttc_refined_latest.pt"
    initial_record = {
        "epoch": 0, "model": model.state_dict(), "validation": baseline, "config": summary,
    }
    atomic_torch_save(initial_record, best_path)
    best_key, history, stale = selection_key(
        baseline, args.general_guard_multiplier
    ), [], 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, teacher, train_loader, optimizer, device, args)
        validation = evaluate(model, validation_loader, device)
        scheduler.step()
        key = selection_key(validation, args.general_guard_multiplier)
        improved = key > best_key
        if improved:
            best_key, stale = key, 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "training": training, "validation": validation,
            "learning_rate": scheduler.get_last_lr()[0], "selection_key": list(key),
        })
        record = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "config": summary,
        }
        atomic_torch_save(record, latest_path)
        if improved:
            atomic_torch_save(record, best_path)
        values = validation["ttc"]
        print(
            f"epoch={epoch:03d} train={training['loss']:.5f} "
            f"ttc={values['inverse_ttc_mae_s_inv']:.5f} "
            f"approach={values['approaching_inverse_ttc_mae_s_inv']:.5f} "
            f"depth={values['inverse_depth_mae_m_inv']:.5f} "
            f"flow={values['flow_epe_cells']:.5f} "
            f"precision={values['critical_precision_at_0_552']:.4f} "
            f"recall={values['critical_recall_at_0_552']:.4f} "
            f"critical_mae={values['critical_inverse_ttc_mae_s_inv']:.5f} "
            f"regression_recall={values['regression_critical_recall']:.4f} "
            f"retained={validation['retention_passed']} improved={improved}", flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch} stale_epochs={stale}", flush=True)
            break

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    final = {
        **summary,
        "best_epoch": int(selected["epoch"]),
        "elapsed_seconds": time.time() - started,
        "validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device, TEST_LIMITS),
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(finite_json(final), indent=2) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(finite_json({
        "best_epoch": final["best_epoch"], "validation": final["validation"],
        "test": final["test"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
