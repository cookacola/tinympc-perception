#!/usr/bin/env python3
"""Mixed fine-tuning of middle-tap gate heads and DroNet scalar navigation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PULP_ROOT = Path("/home/cchen/pulp-dronet/tiny-pulp-dronet-v3")
ISAAC_ROOT = Path("/home/cchen/isaacsim-workspace")
sys.path[:0] = [str(SOURCE_ROOT), str(PULP_ROOT)]

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from fair_espnet_dronetv3_benchmark import (
    evaluate as evaluate_navigation,
    find_strict_pairs,
    make_loader as make_navigation_loader,
    profile_model,
    sample_fingerprint,
    select_f1_threshold,
    seed_everything,
)
from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.data import MultiTaskDataset
from gap8_perception.evaluate import local_centroid
from gap8_perception.losses import soft_dice_loss, weighted_corner_mse
from gap8_perception.model_espnet_dronet_gate import ESPNetDroNetGate


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
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160):
            raise ValueError(self.paths[index])
        return {"image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
                "source": str(self.paths[index])}


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
                if ((corners < 0).any() or (corners[:, 0] >= 160).any()
                        or (corners[:, 1] >= 160).any()):
                    continue
                self.records.append((folder / "stream_out" / row["image"], corners, flight))

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
            maps[channel] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.25**2))
        return {"image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
                "corners": torch.from_numpy(maps), "corner_xy": torch.from_numpy(corners.copy()),
                "corner_valid": torch.tensor(True), "flight": flight, "source": str(path)}


class NavigationView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        return self.model.navigation(frames)


def infinite(loader):
    while True:
        yield from loader


def load_initial_state(model, navigation_checkpoint, gate_checkpoint):
    navigation = torch.load(navigation_checkpoint, map_location="cpu", weights_only=False)
    state = navigation["model"]
    for name in ("stem", "stage1", "stage2", "stage3", "navigation_head"):
        prefix = name + "."
        getattr(model, name).load_state_dict(
            {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
    gate = torch.load(gate_checkpoint, map_location="cpu", weights_only=False)["model"]
    for name in ("corner_adapter", "corner_head", "gate_adapter", "gate_head"):
        prefix = name + "."
        getattr(model, name).load_state_dict(
            {key.removeprefix(prefix): value for key, value in gate.items() if key.startswith(prefix)},
            strict=True,
        )
    nn.init.xavier_uniform_(model.presence_head.weight)
    nn.init.zeros_(model.presence_head.bias)


def gate_forward(model, images):
    # Gate captures are independent stills. Never create false temporal pairs.
    frames = images.repeat(1, 2, 1, 1)
    # Do not let the synthetic/real domain overwrite navigation BatchNorm buffers.
    modules = (model.stem, model.stage1, model.stage2)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    output = model.gate(frames)
    for module, mode in zip(modules, modes):
        module.train(mode)
    return output


def gate_loss(model, synthetic, negative, real, device):
    output = gate_forward(model, synthetic["image"].to(device, non_blocking=True))
    target = synthetic["gate"].to(device, non_blocking=True)
    valid = synthetic["corner_valid"].to(device, non_blocking=True)
    corner = weighted_corner_mse(
        output["corners"], synthetic["corners"].to(device, non_blocking=True), valid
    )
    mask = F.binary_cross_entropy_with_logits(output["gate"], target)
    mask = mask + 0.5 * soft_dice_loss(output["gate"], target)

    negative_output = gate_forward(model, negative["image"].to(device, non_blocking=True))
    negative_mask = F.binary_cross_entropy_with_logits(
        negative_output["gate"], torch.zeros_like(negative_output["gate"])
    )
    presence_logits = torch.cat((output["presence_logit"], negative_output["presence_logit"]))
    presence_labels = torch.cat((
        torch.ones_like(output["presence_logit"]),
        torch.zeros_like(negative_output["presence_logit"]),
    ))
    presence = F.binary_cross_entropy_with_logits(presence_logits, presence_labels)

    real_output = gate_forward(model, real["image"].to(device, non_blocking=True))
    real_corner = weighted_corner_mse(
        real_output["corners"], real["corners"].to(device, non_blocking=True),
        real["corner_valid"].to(device, non_blocking=True),
    )
    real_presence = F.binary_cross_entropy_with_logits(
        real_output["presence_logit"], torch.ones_like(real_output["presence_logit"])
    )
    total = 10 * corner + mask + negative_mask + presence + 5 * real_corner + 0.25 * real_presence
    return total, {
        "corner": corner, "mask": mask, "negative_mask": negative_mask,
        "presence": presence, "real_corner": real_corner, "real_presence": real_presence,
    }


def navigation_loss(model, inputs, labels):
    outputs = model.navigation(inputs)
    yaw = F.mse_loss(outputs[0], labels[:, 0])
    collision = F.binary_cross_entropy(outputs[1].float(), labels[:, 1].float())
    return yaw + collision, yaw, collision


def train_epoch(model, loaders, optimizer, device, navigation_weight):
    model.train()
    streams = {name: infinite(loader) for name, loader in loaders.items()}
    steps = max(len(loaders["navigation"]), len(loaders["synthetic"]))
    totals = {}
    for _ in range(steps):
        inputs, labels = next(streams["navigation"])
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        nav, yaw, collision = navigation_loss(model, inputs, labels)
        auxiliary, parts = gate_loss(
            model, next(streams["synthetic"]), next(streams["negative"]),
            next(streams["real"]), device,
        )
        loss = navigation_weight * nav + auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite mixed loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        values = {"total": loss, "navigation": nav, "yaw": yaw, "collision": collision, **parts}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def gate_raw(model, synthetic_loader, negative_loader, real_loader, device):
    model.eval()
    intersection = union = negative_pixels = negative_total = 0
    errors, real_errors = [], []
    labels, presence, mask_peak, corner_quality = [], [], [], []

    def collect(output, label):
        labels.extend([label] * len(output["presence_logit"]))
        presence.extend(output["presence_logit"].sigmoid().cpu().tolist())
        mask_peak.extend(output["gate"].sigmoid().flatten(1).topk(16, 1).values.mean(1).cpu().tolist())
        corner_quality.extend(output["corners"].sigmoid().flatten(2).amax(2).mean(1).cpu().tolist())

    for batch in synthetic_loader:
        output = gate_forward(model, batch["image"].to(device))
        target = batch["gate"].to(device) >= 0.5
        prediction = output["gate"].sigmoid() >= 0.5
        intersection += int((prediction & target).sum())
        union += int((prediction | target).sum())
        points = local_centroid(output["corners"].sigmoid()) * 4.0
        valid = batch["corner_valid"].to(device)
        if valid.any():
            errors.append(torch.linalg.vector_norm(
                points[valid] - batch["corner_xy"].to(device)[valid], dim=2
            ).cpu())
        collect(output, 1)
    for batch in negative_loader:
        output = gate_forward(model, batch["image"].to(device))
        predicted = output["gate"].sigmoid() >= 0.5
        negative_pixels += int(predicted.sum())
        negative_total += predicted.numel()
        collect(output, 0)
    for batch in real_loader:
        output = gate_forward(model, batch["image"].to(device))
        points = local_centroid(output["corners"].sigmoid()) * 4.0
        real_errors.append(torch.linalg.vector_norm(
            points - batch["corner_xy"].to(device), dim=2
        ).cpu())
        collect(output, 1)
    return {
        "intersection": intersection, "union": union,
        "negative_pixels": negative_pixels, "negative_total": negative_total,
        "errors": torch.cat(errors), "real_errors": torch.cat(real_errors),
        "labels": np.asarray(labels, np.int64),
        "features": np.column_stack((presence, mask_peak, corner_quality)),
    }


def fit_structured_fusion(raw):
    # Tiny deterministic logistic regression; no extra neural deployment graph.
    x = torch.tensor(raw["features"], dtype=torch.float64)
    y = torch.tensor(raw["labels"], dtype=torch.float64)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mean) / scale
    weight = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ weight + bias, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    scores = torch.sigmoid(z @ weight + bias).detach().numpy()
    threshold = select_f1_threshold(raw["labels"], scores)
    return {
        "mean": mean.tolist(), "scale": scale.tolist(),
        "weight": weight.detach().tolist(), "bias": float(bias.detach()),
        "threshold": float(threshold),
    }


def apply_fusion(raw, fusion):
    x = (raw["features"] - np.asarray(fusion["mean"])) / np.asarray(fusion["scale"])
    logits = x @ np.asarray(fusion["weight"]) + fusion["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def binary_metrics(labels, scores, threshold):
    from fair_espnet_dronetv3_benchmark import average_precision, binary_auroc, threshold_metrics
    return {
        "auroc": binary_auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        **threshold_metrics(labels, scores, threshold),
    }


def gate_report(raw, fusion=None):
    error, real_error = raw["errors"], raw["real_errors"]
    report = {
        "gate_iou": raw["intersection"] / max(1, raw["union"]),
        "corner_mean_px": float(error.mean()),
        "corner_p95_px": float(torch.quantile(error, 0.95)),
        "real_corner_mean_px": float(real_error.mean()),
        "real_corner_p95_px": float(torch.quantile(real_error, 0.95)),
        "negative_mask_pixel_rate": raw["negative_pixels"] / max(1, raw["negative_total"]),
    }
    if fusion is not None:
        scores = apply_fusion(raw, fusion)
        report["structured_confidence"] = binary_metrics(raw["labels"], scores, fusion["threshold"])
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "navigation-data", "navigation-checkpoint", "reference-manifest",
        "gate-checkpoint", "gate-dataset", "gate-targets", "gate-split-file",
        "no-gate-dataset", "paired-no-gate-dataset", "real-root", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--gate-batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-5)
    parser.add_argument("--navigation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gate-learning-rate", type=float, default=3e-4)
    parser.add_argument("--navigation-weight", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a Slurm-allocated CUDA GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda")

    samples = {split: find_strict_pairs(args.navigation_data, split) for split in ("train", "valid", "test")}
    reference = json.loads(args.reference_manifest.read_text())
    fingerprints = {split: sample_fingerprint(values) for split, values in samples.items()}
    if fingerprints != reference["fingerprints_sha256"]:
        raise RuntimeError("official PULP-DroNet sample manifest mismatch")

    nav_args = argparse.Namespace(batch_size=args.batch_size, workers=args.workers, seed=args.seed)
    navigation_loaders = {
        split: make_navigation_loader(values, 2, split == "train", nav_args, split == "train")
        for split, values in samples.items()
    }
    synthetic = {
        split: MultiTaskDataset(args.gate_dataset, args.gate_targets, args.gate_split_file, split)
        for split in ("train", "validation", "test")
    }
    base_negative = {
        "train": NoGateDataset(args.no_gate_dataset, (0, 1, 2)),
        "validation": NoGateDataset(args.no_gate_dataset, (3,)),
        "test": NoGateDataset(args.no_gate_dataset, (4,)),
    }
    paired = {"train": range(16), "validation": range(16, 18), "test": range(18, 20)}
    negative = {
        split: ConcatDataset((base_negative[split], NoGateDataset(args.paired_no_gate_dataset, paired[split])))
        for split in base_negative
    }
    real = {
        "train": RealGateDataset(args.real_root, ("flight_06",)),
        "validation": RealGateDataset(args.real_root, ("flight_07",)),
        "test": RealGateDataset(args.real_root, ("flight_08",)),
    }
    def loader(dataset, shuffle=False):
        return DataLoader(
            dataset, args.gate_batch_size, shuffle=shuffle, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
    gate_loaders = {
        split: (loader(synthetic[split]), loader(negative[split]), loader(real[split]))
        for split in ("validation", "test")
    }
    train_loaders = {
        "navigation": navigation_loaders["train"],
        "synthetic": loader(synthetic["train"], True),
        "negative": loader(negative["train"], True),
        "real": loader(real["train"], True),
    }

    model = ESPNetDroNetGate()
    load_initial_state(model, args.navigation_checkpoint, args.gate_checkpoint)
    profile = profile_model(NavigationView(model), 2)
    model.to(device)
    gate_parameters = list(itertools.chain(
        model.corner_adapter.parameters(), model.corner_head.parameters(),
        model.gate_adapter.parameters(), model.gate_head.parameters(), model.presence_head.parameters(),
    ))
    gate_ids = {id(parameter) for parameter in gate_parameters}
    nav_parameters = list(model.navigation_head.parameters())
    nav_ids = {id(parameter) for parameter in nav_parameters}
    encoder_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in gate_ids and id(parameter) not in nav_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_learning_rate},
        {"params": nav_parameters, "lr": args.navigation_learning_rate},
        {"params": gate_parameters, "lr": args.gate_learning_rate},
    ], weight_decay=1e-4)

    best_score, patience = math.inf, args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_loaders, optimizer, device, args.navigation_weight)
        navigation, _, _ = evaluate_navigation(NavigationView(model), navigation_loaders["valid"], device)
        gate_validation = gate_report(gate_raw(model, *gate_loaders["validation"], device))
        score = (
            navigation["yaw_mse"] + navigation["collision_bce"]
            + 0.01 * gate_validation["corner_mean_px"]
            + 0.5 * (1.0 - gate_validation["gate_iou"])
            + 0.25 * gate_validation["negative_mask_pixel_rate"]
        )
        record = {"epoch": epoch, "train": train, "navigation_validation": navigation,
                  "gate_validation": gate_validation, "selection_score": score}
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {"format": "espnet-dronet-middle-gate-v1", "epoch": epoch,
                 "model": model.state_dict(), "record": record, "profile": profile,
                 "tap": "middle_stage2", "corner_head": "heatmap", "gate_head": "binary"}
        torch.save(state, args.output / "last.pt")
        if score < best_score - 1e-4:
            best_score, patience = score, args.patience
            torch.save(state, args.output / "best.pt")
        else:
            patience -= 1
            if patience == 0:
                break

    selected = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    nav_validation, nav_truth, nav_scores = evaluate_navigation(
        NavigationView(model), navigation_loaders["valid"], device
    )
    collision_threshold = select_f1_threshold(nav_truth, nav_scores)
    nav_test, _, _ = evaluate_navigation(
        NavigationView(model), navigation_loaders["test"], device, collision_threshold
    )
    gate_validation_raw = gate_raw(model, *gate_loaders["validation"], device)
    fusion = fit_structured_fusion(gate_validation_raw)
    gate_test_raw = gate_raw(model, *gate_loaders["test"], device)
    summary = {
        "selected_epoch": selected["epoch"], "seed": args.seed,
        "tap": "middle_stage2", "corner_head": "heatmap", "gate_head": "binary",
        "gate_confidence": "logistic fusion of presence, mask peak, and corner quality",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        **profile,
        "sample_fingerprints_sha256": fingerprints,
        "navigation_validation": nav_validation, "navigation_test": nav_test,
        "navigation_collision_threshold": collision_threshold,
        "gate_validation": gate_report(gate_validation_raw, fusion),
        "gate_test": gate_report(gate_test_raw, fusion),
        "structured_confidence_fusion": fusion,
        "real_split": {"train": "flight_06", "validation": "flight_07", "test": "flight_08"},
        "history": history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
