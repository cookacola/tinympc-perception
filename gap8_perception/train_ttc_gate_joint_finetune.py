#!/usr/bin/env python3
"""Phase-B gate fine-tuning with narrow e2 adaptation and TTC retention gates."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .train_ttc_gate_head import (
    atomic_torch_save,
    empty_metrics,
    finalize_metrics,
    finite_json,
    seed_everything,
    update_metrics,
)
from .ttc_gate_data import TTCGateDataset, gate_sampling_weights
from .ttc_gate_losses import gate_perception_loss
from .ttc_motion_gate_model import (
    MotionConditionedESPNetGateTTCNet,
    MotionConditionedESPNetInverseTTCNet,
)
from .ttc_motion_losses import (
    motion_conditioned_ttc_loss,
    parent_distillation_loss,
    risk_class,
)


PARENT_KEYS = (
    "images", "onboard_state", "inverse_ttc", "ttc_valid", "ttc_approaching",
    "inverse_depth", "depth_valid", "flow", "flow_valid",
    "gate_corners_px", "gate_corners_visible",
    "gate_supervision_eligible",
)
VALIDATION_LIMITS = {
    "inverse_ttc_mae_s_inv": 0.16806538474782983,
    "approaching_inverse_ttc_mae_s_inv": 0.18358012907349132,
    "inverse_depth_mae_m_inv": 0.23257946326150257,
    "flow_epe_cells": 0.1336182888123062,
    "critical_precision_at_0_552": 0.695,
    "critical_recall_at_0_552": 0.734,
}
TEST_LIMITS = {
    "inverse_ttc_mae_s_inv": 0.15150342336117208,
    "approaching_inverse_ttc_mae_s_inv": 0.16786268978708097,
    "inverse_depth_mae_m_inv": 0.19982632630495535,
    "flow_epe_cells": 0.12236654571224394,
    "critical_precision_at_0_552": 0.6804023741992432,
    "critical_recall_at_0_552": 0.7347273707953319,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-gate-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--gate-learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=2.0)
    parser.add_argument("--supervised-ttc-weight", type=float, default=0.25)
    parser.add_argument("--minimum-pck8-gain", type=float, default=0.02)
    parser.add_argument(
        "--encoder-scope", choices=("last_e2", "all_mid"), default="last_e2"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--maximum-gate-distance-m", type=float, default=8.0)
    parser.add_argument("--minimum-gate-span-px", type=float, default=16.0)
    parser.add_argument("--minimum-gate-area-px2", type=float, default=256.0)
    return parser.parse_args()


def move_batch(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in PARENT_KEYS}


def empty_ttc_metrics():
    return {
        "valid": 0, "approaching": 0, "depth_valid": 0, "flow_valid": 0,
        "ttc_abs": 0.0, "approaching_abs": 0.0, "depth_abs": 0.0, "flow_epe": 0.0,
        "critical_tp": 0, "critical_fp": 0, "critical_fn": 0,
        "regression_critical_tp": 0, "regression_critical_fp": 0,
        "regression_critical_fn": 0, "critical_ttc_abs": 0.0,
        "critical_ttc_pixels": 0,
    }


def update_ttc_metrics(accumulator, prediction, target):
    valid = target["ttc_valid"].bool()
    approaching = target["ttc_approaching"].bool() & valid
    depth_valid = target["depth_valid"].bool()
    flow_valid = target["flow_valid"].bool()
    ttc_error = (prediction["inverse_ttc"] - target["inverse_ttc"]).abs()
    depth_error = (prediction["inverse_depth"] - target["inverse_depth"]).abs()
    flow_error = torch.sqrt(
        ((prediction["flow"] - target["flow"]) ** 2).sum(1, keepdim=True) + 1e-6
    )
    truth_critical = (
        (risk_class(target["inverse_ttc"]) == 2)
        & target["ttc_approaching"].bool() & valid
    ).squeeze(1)
    predicted_critical = torch.softmax(prediction["risk_logits"], 1)[:, 2] >= 0.552
    regression_critical = prediction["inverse_ttc"].squeeze(1) >= 2.0
    valid_2d = valid.squeeze(1)
    accumulator["valid"] += int(valid.sum())
    accumulator["approaching"] += int(approaching.sum())
    accumulator["depth_valid"] += int(depth_valid.sum())
    accumulator["flow_valid"] += int(flow_valid.sum())
    accumulator["ttc_abs"] += float((ttc_error * valid).sum())
    accumulator["approaching_abs"] += float((ttc_error * approaching).sum())
    accumulator["depth_abs"] += float((depth_error * depth_valid).sum())
    accumulator["flow_epe"] += float((flow_error * flow_valid).sum())
    accumulator["critical_tp"] += int((predicted_critical & truth_critical).sum())
    accumulator["critical_fp"] += int((predicted_critical & ~truth_critical & valid_2d).sum())
    accumulator["critical_fn"] += int((~predicted_critical & truth_critical).sum())
    accumulator["regression_critical_tp"] += int(
        (regression_critical & truth_critical).sum()
    )
    accumulator["regression_critical_fp"] += int(
        (regression_critical & ~truth_critical & valid_2d).sum()
    )
    accumulator["regression_critical_fn"] += int(
        (~regression_critical & truth_critical).sum()
    )
    accumulator["critical_ttc_abs"] += float(
        (ttc_error.squeeze(1) * truth_critical).sum()
    )
    accumulator["critical_ttc_pixels"] += int(truth_critical.sum())


def finalize_ttc_metrics(accumulator):
    tp, fp, fn = (
        accumulator["critical_tp"], accumulator["critical_fp"], accumulator["critical_fn"]
    )
    regression_tp, regression_fp, regression_fn = (
        accumulator["regression_critical_tp"],
        accumulator["regression_critical_fp"],
        accumulator["regression_critical_fn"],
    )
    return {
        "inverse_ttc_mae_s_inv": accumulator["ttc_abs"] / max(accumulator["valid"], 1),
        "approaching_inverse_ttc_mae_s_inv": (
            accumulator["approaching_abs"] / max(accumulator["approaching"], 1)
        ),
        "inverse_depth_mae_m_inv": (
            accumulator["depth_abs"] / max(accumulator["depth_valid"], 1)
        ),
        "flow_epe_cells": accumulator["flow_epe"] / max(accumulator["flow_valid"], 1),
        "critical_precision_at_0_552": tp / max(tp + fp, 1),
        "critical_recall_at_0_552": tp / max(tp + fn, 1),
        "critical_counts_at_0_552": {"tp": tp, "fp": fp, "fn": fn},
        "critical_inverse_ttc_mae_s_inv": (
            accumulator["critical_ttc_abs"] / max(accumulator["critical_ttc_pixels"], 1)
        ),
        "regression_critical_precision": (
            regression_tp / max(regression_tp + regression_fp, 1)
        ),
        "regression_critical_recall": (
            regression_tp / max(regression_tp + regression_fn, 1)
        ),
        "regression_critical_counts": {
            "tp": regression_tp, "fp": regression_fp, "fn": regression_fn,
        },
    }


def retention_passes(metrics, limits=VALIDATION_LIMITS):
    return all(
        metrics[name] <= limit if name not in {
            "critical_precision_at_0_552", "critical_recall_at_0_552"
        } else metrics[name] >= limit
        for name, limit in limits.items()
    )


def evaluate(model, loader, device, retention_limits=VALIDATION_LIMITS):
    model.eval()
    gate, ttc = empty_metrics(), empty_ttc_metrics()
    with torch.no_grad():
        for raw in loader:
            target = move_batch(raw, device)
            prediction = model(target["images"], target["onboard_state"])
            gate_loss, parts = gate_perception_loss(prediction, target)
            update_metrics(gate, prediction, target, gate_loss, parts)
            update_ttc_metrics(ttc, prediction, target)
    ttc_result = finalize_ttc_metrics(ttc)
    return {
        "gate": finalize_metrics(gate),
        "ttc": ttc_result,
        "retention_passed": retention_passes(ttc_result, retention_limits),
        "retention_limits": retention_limits,
    }


def configure_trainable(model, encoder_scope="last_e2"):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.gate_decoder.parameters():
        parameter.requires_grad_(True)
    if encoder_scope == "last_e2":
        encoder_modules = (model.encoder.stage2[1],)
    elif encoder_scope == "all_mid":
        encoder_modules = (model.encoder.stem, model.encoder.stage1, model.encoder.stage2)
    else:
        raise ValueError(f"unknown encoder scope {encoder_scope}")
    encoder_parameters = []
    for module in encoder_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            encoder_parameters.append(parameter)
    return list(model.gate_decoder.parameters()), encoder_parameters


def joint_training_mode(model):
    model.eval()
    model.gate_decoder.train()


def train_epoch(model, teacher, loader, optimizer, device, kd_weight, ttc_weight):
    joint_training_mode(model)
    total, examples = 0.0, 0
    for raw in loader:
        target = move_batch(raw, device)
        with torch.no_grad():
            teacher_output = teacher(target["images"], target["onboard_state"])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(target["images"], target["onboard_state"])
        gate, _gate_parts = gate_perception_loss(prediction, target)
        distillation, _kd_parts = parent_distillation_loss(prediction, teacher_output)
        ttc, _ttc_parts = motion_conditioned_ttc_loss(prediction, target)
        loss = gate + kd_weight * distillation + ttc_weight * ttc
        loss.backward()
        optimizer.step()
        count = target["images"].shape[0]
        total += float(loss.detach()) * count
        examples += count
    return total / max(examples, 1)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("joint gate fine-tuning requires a Slurm GPU allocation")
    quality = dict(
        maximum_gate_distance_m=args.maximum_gate_distance_m,
        minimum_gate_span_px=args.minimum_gate_span_px,
        minimum_gate_area_px2=args.minimum_gate_area_px2,
    )
    train_set = TTCGateDataset(args.dataset, "train", augment=True, **quality)
    validation_set = TTCGateDataset(args.dataset, "validation", **quality)
    test_set = TTCGateDataset(args.dataset, "test", **quality)
    weights, sampling = gate_sampling_weights(train_set)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=sampler, **options)
    validation_loader = DataLoader(validation_set, shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)

    model = MotionConditionedESPNetGateTTCNet().to(device)
    initialization = model.initialize_from_checkpoint(args.frozen_gate_checkpoint)
    if initialization["mode"] != "full_gate_checkpoint":
        raise RuntimeError("phase B requires a complete frozen-head checkpoint")
    teacher = MotionConditionedESPNetInverseTTCNet().to(device)
    teacher.initialize_from_checkpoint(args.parent_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    gate_parameters, encoder_parameters = configure_trainable(model, args.encoder_scope)
    optimizer = torch.optim.AdamW([
        {"params": gate_parameters, "lr": args.gate_learning_rate},
        {"params": encoder_parameters, "lr": args.encoder_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    baseline = evaluate(model, validation_loader, device)
    if not baseline["retention_passed"]:
        raise RuntimeError(f"frozen phase-A checkpoint failed retention limits: {baseline['ttc']}")
    best_path = args.output_dir / "espnet_motion_gate_ttc_joint_best.pt"
    latest_path = args.output_dir / "espnet_motion_gate_ttc_joint_latest.pt"
    initial_record = {
        "epoch": 0, "model": model.state_dict(), "validation": baseline,
        "initialization": initialization,
    }
    atomic_torch_save(initial_record, best_path)
    best_pck8 = baseline["gate"]["pck8"]
    history, stale = [], 0
    started = time.time()
    summary = {
        "experiment": "ttc_motion_gate_joint_finetune_v1",
        "dataset": str(args.dataset.resolve()),
        "gate_supervision_policy": train_set.gate_supervision_policy(),
        "sampling": sampling,
        "initialization": initialization,
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "trainable": ["gate_decoder", args.encoder_scope],
        "encoder_scope": args.encoder_scope,
        "trainable_parameters": sum(parameter.numel() for parameter in gate_parameters + encoder_parameters),
        "gate_learning_rate": args.gate_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "distillation_weight": args.distillation_weight,
        "supervised_ttc_weight": args.supervised_ttc_weight,
        "validation_limits": VALIDATION_LIMITS,
        "test_limits": TEST_LIMITS,
        "minimum_pck8_gain": args.minimum_pck8_gain,
        "phase_a_validation": baseline,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, teacher, train_loader, optimizer, device,
            args.distillation_weight, args.supervised_ttc_weight,
        )
        validation = evaluate(model, validation_loader, device)
        scheduler.step()
        improved = validation["retention_passed"] and validation["gate"]["pck8"] > best_pck8
        if improved:
            best_pck8, stale = validation["gate"]["pck8"], 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "learning_rates": scheduler.get_last_lr(), "validation": validation,
            "improved": improved,
        })
        record = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "config": summary,
        }
        atomic_torch_save(record, latest_path)
        if improved:
            atomic_torch_save(record, best_path)
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} "
            f"pck8={validation['gate']['pck8']:.4f} "
            f"mae_px={validation['gate']['corner_mae_px']:.3f} "
            f"vis_f1={validation['gate']['visibility_f1']:.4f} "
            f"ttc_mae={validation['ttc']['inverse_ttc_mae_s_inv']:.5f} "
            f"retained={validation['retention_passed']} improved={improved}", flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch} stale_epochs={stale}", flush=True)
            break

    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    final_validation = evaluate(model, validation_loader, device)
    gain = final_validation["gate"]["pck8"] - baseline["gate"]["pck8"]
    promoted = bool(saved["epoch"] > 0 and gain >= args.minimum_pck8_gain)
    final = {
        **summary,
        "best_epoch": int(saved["epoch"]),
        "elapsed_seconds": time.time() - started,
        "validation": final_validation,
        "validation_pck8_gain": gain,
        "promoted_over_phase_a": promoted,
        "test": evaluate(model, test_loader, device, TEST_LIMITS) if promoted else None,
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(finite_json(final), indent=2) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(finite_json({
        "best_epoch": final["best_epoch"], "validation_pck8_gain": gain,
        "promoted_over_phase_a": promoted, "validation": final_validation,
        "test": final["test"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
