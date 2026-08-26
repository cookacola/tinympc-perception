#!/usr/bin/env python3
"""Retrain both heads of the stock-DORY-partitioned gate/TTC network."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .train_ttc_gate_head import atomic_torch_save, finite_json, seed_everything
from .train_ttc_gate_joint_finetune import evaluate, move_batch
from .ttc_gate_data import TTCGateDataset, gate_sampling_weights
from .ttc_gate_losses import gate_perception_loss
from .ttc_motion_gate_dory_model import DoryPartitionedMotionGateTTCNet
from .ttc_motion_gate_model import MotionConditionedESPNetInverseTTCNet
from .ttc_motion_losses import motion_conditioned_ttc_loss, parent_distillation_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-bridge", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-loss-weight", type=float, default=0.25)
    parser.add_argument("--ttc-loss-weight", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--gate-balanced-fraction", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--maximum-gate-distance-m", type=float, default=8.0)
    parser.add_argument("--minimum-gate-span-px", type=float, default=16.0)
    parser.add_argument("--minimum-gate-area-px2", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def train_epoch(model, teacher, loader, optimizer, device, args):
    model.train()
    teacher.eval()
    totals = {"loss": 0.0, "gate": 0.0, "ttc": 0.0, "distillation": 0.0}
    examples = 0
    for raw in loader:
        target = move_batch(raw, device)
        with torch.no_grad():
            teacher_output = teacher(target["images"], target["onboard_state"])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(target["images"], target["onboard_state"])
        gate, _gate_parts = gate_perception_loss(prediction, target)
        ttc, _ttc_parts = motion_conditioned_ttc_loss(prediction, target)
        distillation, _distill_parts = parent_distillation_loss(
            prediction, teacher_output
        )
        loss = (
            args.gate_loss_weight * gate
            + args.ttc_loss_weight * ttc
            + args.distillation_weight * distillation
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count = target["images"].shape[0]
        examples += count
        for name, value in (
            ("loss", loss), ("gate", gate), ("ttc", ttc),
            ("distillation", distillation),
        ):
            totals[name] += float(value.detach()) * count
    return {name: value / max(examples, 1) for name, value in totals.items()}


def selection_key(validation):
    gate, ttc = validation["gate"], validation["ttc"]
    return (
        int(validation["retention_passed"]),
        gate["pck8"],
        gate["visibility_f1"],
        -ttc["inverse_ttc_mae_s_inv"],
        -ttc["approaching_inverse_ttc_mae_s_inv"],
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DORY gate/TTC training requires a Slurm GPU allocation")

    quality = dict(
        maximum_gate_distance_m=args.maximum_gate_distance_m,
        minimum_gate_span_px=args.minimum_gate_span_px,
        minimum_gate_area_px2=args.minimum_gate_area_px2,
    )
    train_set = TTCGateDataset(args.dataset, "train", augment=True, **quality)
    validation_set = TTCGateDataset(args.dataset, "validation", **quality)
    test_set = TTCGateDataset(args.dataset, "test", **quality)
    gate_weights, sampling = gate_sampling_weights(train_set)
    natural = torch.full_like(gate_weights, 1.0 / len(train_set))
    weights = (
        args.gate_balanced_fraction * gate_weights
        + (1.0 - args.gate_balanced_fraction) * natural
    )
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

    model = DoryPartitionedMotionGateTTCNet().to(device)
    initialization = model.initialize_from_dory_bridge(args.initialization_bridge)
    teacher = MotionConditionedESPNetInverseTTCNet().to(device)
    teacher_initialization = teacher.initialize_from_checkpoint(args.teacher_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
        {"params": model.gate_head.parameters(), "lr": args.head_learning_rate},
        {"params": model.ttc_head.parameters(), "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    summary = {
        "experiment": "dory_partitioned_motion_gate_ttc_v1",
        "architecture": {
            "graphs": {
                "encoder": {"input": [2, 160, 160], "output": [64, 20, 20]},
                "gate_head": {"input": [64, 20, 20], "output": [8, 20, 20]},
                "ttc_head": {"input": [74, 20, 20], "output": [7, 20, 20]},
            },
            "learned_operator_contract": ["Conv", "Relu", "Add"],
            "gate_output_channels": [
                "corner_tl", "corner_tr", "corner_br", "corner_bl",
                "visibility_tl", "visibility_tr", "visibility_br", "visibility_bl",
            ],
            "ttc_output_channels": [
                "inverse_ttc", "inverse_depth", "flow_x", "flow_y",
                "risk_safe", "risk_warning", "risk_critical",
            ],
            "state_packing": "10 normalized onboard scalars broadcast beside e2",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": str(args.dataset.resolve()),
        "splits": {
            "train": len(train_set), "validation": len(validation_set), "test": len(test_set),
        },
        "gate_supervision_policy": train_set.gate_supervision_policy(),
        "sampling": {
            **sampling,
            "gate_balanced_fraction": args.gate_balanced_fraction,
            "natural_fraction": 1.0 - args.gate_balanced_fraction,
        },
        "initialization": initialization,
        "teacher_initialization": teacher_initialization,
        "teacher_is_training_only": True,
        "loss_weights": {
            "gate": args.gate_loss_weight, "ttc": args.ttc_loss_weight,
            "distillation": args.distillation_weight,
        },
        "head_learning_rate": args.head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "seed": args.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(summary, indent=2), flush=True)

    best_path = args.output_dir / "dory_motion_gate_ttc_best.pt"
    latest_path = args.output_dir / "dory_motion_gate_ttc_latest.pt"
    history, best_key, stale = [], None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, teacher, train_loader, optimizer, device, args)
        validation = evaluate(model, validation_loader, device)
        scheduler.step()
        key = selection_key(validation)
        improved = best_key is None or key > best_key
        if improved:
            best_key, stale = key, 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "training": training, "validation": validation,
            "learning_rates": scheduler.get_last_lr(), "selection_key": list(key),
        })
        record = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "config": summary,
        }
        atomic_torch_save(record, latest_path)
        if improved:
            atomic_torch_save(record, best_path)
        print(
            f"epoch={epoch:03d} train={training['loss']:.5f} "
            f"pck8={validation['gate']['pck8']:.4f} "
            f"vis_f1={validation['gate']['visibility_f1']:.4f} "
            f"ttc={validation['ttc']['inverse_ttc_mae_s_inv']:.5f} "
            f"critical_recall={validation['ttc']['critical_recall_at_0_552']:.4f} "
            f"retained={validation['retention_passed']} improved={improved}", flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch} stale_epochs={stale}", flush=True)
            break

    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    final = {
        **summary,
        "best_epoch": int(saved["epoch"]),
        "elapsed_seconds": time.time() - started,
        "validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device),
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
