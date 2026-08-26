#!/usr/bin/env python3
"""Train the mid-level four-heatmap and per-corner visibility gate branch."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .ttc_gate_data import TTCGateDataset, gate_sampling_weights
from .ttc_gate_losses import (
    gate_perception_loss,
    peak_gate_coordinates,
    softargmax_gate_coordinates,
)
from .ttc_motion_gate_model import (
    MotionConditionedESPNetGateTTCNet,
    MotionConditionedESPNetInverseTTCNet,
)


DEFAULT_CHECKPOINT = Path("releases/ttc_motion_v1/model/ttc_motion_v1_epoch20.pt")
PARENT_OUTPUTS = ("inverse_ttc", "inverse_depth", "flow", "risk_logits")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def move_batch(batch, device):
    keys = (
        "images", "onboard_state", "gate_corners_px", "gate_corners_visible",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def empty_metrics():
    return {
        "examples": 0,
        "loss_sum": 0.0,
        "parts": {},
        "visible_corners": 0,
        "error_sum_px": 0.0,
        "softargmax_error_sum_px": 0.0,
        "pck4": 0,
        "pck8": 0,
        "pck12": 0,
        "visibility_tp": 0,
        "visibility_fp": 0,
        "visibility_tn": 0,
        "visibility_fn": 0,
        "all_four_frames": 0,
        "all_four_pck8": 0,
        "by_visible_count": {
            str(index): {"frames": 0, "visible": 0, "error_sum_px": 0.0, "pck8": 0}
            for index in range(5)
        },
    }


def update_metrics(accumulator, prediction, target, loss, parts):
    batch_size = target["images"].shape[0]
    accumulator["examples"] += batch_size
    accumulator["loss_sum"] += float(loss) * batch_size
    for name, value in parts.items():
        accumulator["parts"].setdefault(name, 0.0)
        accumulator["parts"][name] += float(value) * batch_size
    coordinates = peak_gate_coordinates(prediction["gate_heatmap_logits"])
    soft_coordinates = softargmax_gate_coordinates(prediction["gate_heatmap_logits"])
    error = torch.linalg.vector_norm(coordinates - target["gate_corners_px"], dim=-1)
    soft_error = torch.linalg.vector_norm(
        soft_coordinates - target["gate_corners_px"], dim=-1
    )
    visible = target["gate_corners_visible"].bool()
    predicted_visible = prediction["gate_visibility_logits"] >= 0.0
    accumulator["visible_corners"] += int(visible.sum())
    accumulator["error_sum_px"] += float(error[visible].sum())
    accumulator["softargmax_error_sum_px"] += float(soft_error[visible].sum())
    accumulator["pck4"] += int(((error <= 4.0) & visible).sum())
    accumulator["pck8"] += int(((error <= 8.0) & visible).sum())
    accumulator["pck12"] += int(((error <= 12.0) & visible).sum())
    accumulator["visibility_tp"] += int((predicted_visible & visible).sum())
    accumulator["visibility_fp"] += int((predicted_visible & ~visible).sum())
    accumulator["visibility_tn"] += int((~predicted_visible & ~visible).sum())
    accumulator["visibility_fn"] += int((~predicted_visible & visible).sum())
    visible_count = visible.sum(1)
    all_four = visible_count == 4
    accumulator["all_four_frames"] += int(all_four.sum())
    accumulator["all_four_pck8"] += int((all_four & (error <= 8.0).all(1)).sum())
    for count in range(5):
        selected = visible_count == count
        group = accumulator["by_visible_count"][str(count)]
        group["frames"] += int(selected.sum())
        group["visible"] += int(visible[selected].sum())
        group["error_sum_px"] += float(error[selected][visible[selected]].sum())
        group["pck8"] += int(((error[selected] <= 8.0) & visible[selected]).sum())


def finalize_metrics(accumulator):
    examples = max(accumulator["examples"], 1)
    visible = max(accumulator["visible_corners"], 1)
    tp, fp = accumulator["visibility_tp"], accumulator["visibility_fp"]
    tn, fn = accumulator["visibility_tn"], accumulator["visibility_fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    by_count = {}
    for count, group in accumulator["by_visible_count"].items():
        group_visible = max(group["visible"], 1)
        by_count[count] = {
            "frames": group["frames"],
            "visible_corners": group["visible"],
            "corner_mae_px": group["error_sum_px"] / group_visible,
            "pck8": group["pck8"] / group_visible,
        }
    return {
        "examples": accumulator["examples"],
        "loss": accumulator["loss_sum"] / examples,
        **{name: value / examples for name, value in accumulator["parts"].items()},
        "visible_corners": accumulator["visible_corners"],
        "corner_mae_px": accumulator["error_sum_px"] / visible,
        "global_softargmax_corner_mae_px": (
            accumulator["softargmax_error_sum_px"] / visible
        ),
        "pck4": accumulator["pck4"] / visible,
        "pck8": accumulator["pck8"] / visible,
        "pck12": accumulator["pck12"] / visible,
        "all_four_pck8": accumulator["all_four_pck8"] / max(accumulator["all_four_frames"], 1),
        "all_four_frames": accumulator["all_four_frames"],
        "visibility_accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "visibility_precision": precision,
        "visibility_recall": recall,
        "visibility_specificity": specificity,
        "visibility_balanced_accuracy": 0.5 * (recall + specificity),
        "visibility_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "visibility_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_visible_count": by_count,
    }


def evaluate(model, loader, device):
    model.eval()
    accumulator = empty_metrics()
    with torch.no_grad():
        for raw in loader:
            target = move_batch(raw, device)
            prediction = model(target["images"], target["onboard_state"])
            loss, parts = gate_perception_loss(prediction, target)
            update_metrics(accumulator, prediction, target, loss, parts)
    return finalize_metrics(accumulator)


def train_epoch(model, loader, optimizer, device):
    model.gate_training_mode()
    total, examples = 0.0, 0
    for raw in loader:
        target = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(target["images"], target["onboard_state"])
        loss, _parts = gate_perception_loss(prediction, target)
        loss.backward()
        optimizer.step()
        count = target["images"].shape[0]
        total += float(loss.detach()) * count
        examples += count
    return total / max(examples, 1)


def selection_key(metrics):
    return (
        metrics["pck8"], metrics["visibility_f1"],
        -metrics["corner_mae_px"], -metrics["loss"],
    )


def parent_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("gate_decoder.")
    }


def compare_parent_state(model, reference):
    current = parent_state(model)
    changed = [name for name in reference if not torch.equal(reference[name], current[name])]
    return {"exact": not changed, "changed_tensors": changed}


def compare_parent_outputs(model, checkpoint, batch, device):
    baseline = MotionConditionedESPNetInverseTTCNet().to(device)
    baseline.initialize_from_checkpoint(checkpoint)
    baseline.eval()
    model.eval()
    with torch.no_grad():
        base = baseline(batch["images"], batch["onboard_state"])
        joint = model(batch["images"], batch["onboard_state"])
    differences = {
        name: float((base[name] - joint[name]).abs().max()) for name in PARENT_OUTPUTS
    }
    return {"exact": all(value == 0.0 for value in differences.values()), "max_abs": differences}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("gate-head training requires a Slurm GPU allocation")

    train_set = TTCGateDataset(args.dataset, "train", augment=True)
    validation_set = TTCGateDataset(args.dataset, "validation")
    test_set = TTCGateDataset(args.dataset, "test")
    sampler_weights, sampling = gate_sampling_weights(train_set)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sampler_weights, num_samples=len(train_set), replacement=True, generator=generator
    )
    loader_options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model = MotionConditionedESPNetGateTTCNet().to(device)
    initialization = model.initialize_from_checkpoint(args.initial_checkpoint)
    if initialization["mode"] != "v1_ttc_plus_fresh_gate":
        raise RuntimeError("this experiment requires the gate-free v1 parent checkpoint")
    model.freeze_parent_for_gate_training()
    initial_parent = parent_state(model)
    first = move_batch(next(iter(validation_loader)), device)
    initial_output_retention = compare_parent_outputs(
        model, args.initial_checkpoint, first, device
    )
    if not initial_output_retention["exact"]:
        raise RuntimeError(f"augmented model changed v1 outputs: {initial_output_retention}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    summary = {
        "experiment": "ttc_motion_gate_head_v1",
        "architecture": {
            "gate_branch_input": "encoder.e2",
            "gate_branch_input_shape": [64, 20, 20],
            "corner_order": list(model.gate_corner_order),
            "heatmap_shape": [4, 20, 20],
            "visibility_shape": [4],
            "visibility_semantics": model.gate_visibility_semantics,
            "parent_frozen": True,
        },
        "dataset": str(args.dataset.resolve()),
        "splits": {
            "train": len(train_set), "validation": len(validation_set), "test": len(test_set)
        },
        "sampling": sampling,
        "initialization": initialization,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_parent_output_retention": initial_output_retention,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(summary, indent=2), flush=True)

    best_path = args.output_dir / "espnet_motion_gate_ttc_best.pt"
    latest_path = args.output_dir / "espnet_motion_gate_ttc_latest.pt"
    history, best_key, stale = [], None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        scheduler.step()
        key = selection_key(validation)
        improved = best_key is None or key > best_key
        stale = 0 if improved else stale + 1
        if improved:
            best_key = key
        history.append({
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "validation": validation,
            "selection_key": list(key),
        })
        record = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "config": summary,
        }
        atomic_torch_save(record, latest_path)
        if improved:
            atomic_torch_save(record, best_path)
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={validation['loss']:.6f} "
            f"mae_px={validation['corner_mae_px']:.3f} pck8={validation['pck8']:.4f} "
            f"vis_f1={validation['visibility_f1']:.4f} improved={improved}",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch} stale_epochs={stale}", flush=True)
            break

    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    final_parent_state = compare_parent_state(model, initial_parent)
    final_parent_outputs = compare_parent_outputs(model, args.initial_checkpoint, first, device)
    if not final_parent_state["exact"] or not final_parent_outputs["exact"]:
        raise RuntimeError("frozen v1 parent did not remain bit-exact")
    final = {
        **summary,
        "best_epoch": int(saved["epoch"]),
        "elapsed_seconds": time.time() - started,
        "validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device),
        "parent_state_retention": final_parent_state,
        "parent_output_retention": final_parent_outputs,
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(finite_json(final), indent=2) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(finite_json({
        "best_epoch": final["best_epoch"], "validation": final["validation"],
        "test": final["test"], "parent_output_retention": final["parent_output_retention"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
