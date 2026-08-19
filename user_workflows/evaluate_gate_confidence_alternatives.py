#!/usr/bin/env python3
"""Compare leakage-safe gate-confidence scores on a frozen mixed model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ISAACSIM_REPO = Path("/home/cchen/isaacsim-workspace")
sys.path.insert(0, str(ISAACSIM_REPO))

from gap8_perception.data import MultiTaskDataset
from gap8_perception.evaluate import local_centroid
from train_retained_obstacle_gate import (
    NoGateDataset,
    RealGateDataset,
    RetainedObstacleGateModel,
)


FEATURE_NAMES = (
    "presence_logit",
    "mask_mean",
    "mask_max",
    "mask_top1pct",
    "mask_area_050",
    "mask_area_025",
    "mask_largest_component",
    "corner_peak_mean",
    "corner_peak_min",
    "corner_contrast_mean",
    "corner_sharpness_mean",
    "corner_mask_agreement_mean",
    "corner_mask_agreement_min",
    "corner_bbox_area",
    "corner_min_separation",
)


def largest_component_fraction(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.float32)
    for index, binary in enumerate(mask):
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        if count > 1:
            result[index] = float(stats[1:, cv2.CC_STAT_AREA].max()) / binary.size
    return result


@torch.no_grad()
def batch_features(output: dict[str, torch.Tensor]) -> np.ndarray:
    mask = output["gate"].sigmoid()
    corners = output["corners"].sigmoid()
    batch = mask.shape[0]
    flat_mask = mask.flatten(1)
    flat_corners = corners.flatten(2)

    mask_mean = flat_mask.mean(1)
    mask_max = flat_mask.max(1).values
    mask_top = flat_mask.topk(max(1, flat_mask.shape[1] // 100), dim=1).values.mean(1)
    area_050 = (flat_mask >= 0.50).float().mean(1)
    area_025 = (flat_mask >= 0.25).float().mean(1)
    component = torch.from_numpy(
        largest_component_fraction((mask[:, 0] >= 0.50).cpu().numpy())
    ).to(mask.device)

    corner_peak = flat_corners.max(2).values
    corner_mean = flat_corners.mean(2)
    contrast = corner_peak - corner_mean
    normalized = flat_corners / flat_corners.sum(2, keepdim=True).clamp_min(1e-8)
    entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(2)
    sharpness = 1.0 - entropy / math.log(flat_corners.shape[2])

    points = local_centroid(corners)
    point_index = points.round().long()
    point_x = point_index[..., 0].clamp(0, mask.shape[3] - 1)
    point_y = point_index[..., 1].clamp(0, mask.shape[2] - 1)
    batch_index = torch.arange(batch, device=mask.device)[:, None]
    agreement = mask[batch_index, 0, point_y, point_x]

    x_span = (points[..., 0].max(1).values - points[..., 0].min(1).values) / mask.shape[3]
    y_span = (points[..., 1].max(1).values - points[..., 1].min(1).values) / mask.shape[2]
    bbox_area = x_span * y_span
    distances = torch.cdist(points, points)
    distances += torch.eye(4, device=mask.device)[None] * 1e6
    min_separation = distances.flatten(1).min(1).values / math.hypot(mask.shape[2], mask.shape[3])

    columns = (
        output["presence_logit"],
        mask_mean,
        mask_max,
        mask_top,
        area_050,
        area_025,
        component,
        corner_peak.mean(1),
        corner_peak.min(1).values,
        contrast.mean(1),
        sharpness.mean(1),
        agreement.mean(1),
        agreement.min(1).values,
        bbox_area,
        min_separation,
    )
    return torch.stack(columns, 1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def collect(model, loader, device, domain: str) -> dict[str, np.ndarray]:
    features = []
    labels = []
    domains = []
    sources = []
    for batch in loader:
        output = model.forward_gate(batch["image"].to(device, non_blocking=True))
        values = batch_features(output)
        features.append(values)
        if domain == "synthetic":
            visible = batch["gate"].flatten(1).any(1).numpy().astype(bool)
            labels.append(visible.astype(np.int8))
            domains.extend([
                "synthetic_visible" if is_visible else "synthetic_absent"
                for is_visible in visible
            ])
        else:
            labels.append(np.full(len(values), domain != "negative", dtype=np.int8))
            domains.extend([domain] * len(values))
        raw_sources = batch.get("source", [""] * len(values))
        sources.extend([str(source) for source in raw_sources])
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "domains": np.asarray(domains),
        "sources": np.asarray(sources),
    }


def merge(*parts) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    balanced = (tpr + 1.0 - fpr) / 2.0
    best_value = balanced.max()
    candidates = np.flatnonzero(np.isclose(balanced, best_value))
    best = candidates[np.argmin(fpr[candidates])]
    return float(thresholds[best]), float(best_value)


def metrics(data: dict[str, np.ndarray], scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = data["labels"].astype(bool)
    domains = data["domains"]
    prediction = scores >= threshold
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    result = {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float((prediction == labels).mean()),
        "bce": float(log_loss(labels, clipped)),
        "brier": float(brier_score_loss(labels, clipped)),
        "threshold": float(threshold),
    }
    for domain in ("synthetic_visible", "synthetic_absent", "real", "negative"):
        chosen = domains == domain
        if chosen.any():
            key = "false_positive_rate" if domain in {"negative", "synthetic_absent"} else "recall"
            result[f"{domain}_{key}"] = float(prediction[chosen].mean())
            result[f"{domain}_mean_score"] = float(scores[chosen].mean())
            result[f"{domain}_examples"] = int(chosen.sum())
    return result


def direct_scores(features: np.ndarray) -> dict[str, np.ndarray]:
    f = {name: features[:, index] for index, name in enumerate(FEATURE_NAMES)}
    presence = 1.0 / (1.0 + np.exp(-np.clip(f["presence_logit"], -30, 30)))
    heatmap_quality = np.sqrt(
        np.clip(f["corner_peak_min"], 0, 1) * np.clip(f["corner_sharpness_mean"], 0, 1)
    )
    mask_corner_joint = np.cbrt(
        np.clip(f["mask_top1pct"], 0, 1)
        * np.clip(f["corner_mask_agreement_mean"], 0, 1)
        * np.clip(heatmap_quality, 0, 1)
    )
    return {
        "standalone_presence_head": presence,
        "mask_top1pct": f["mask_top1pct"],
        "mask_largest_component": f["mask_largest_component"],
        "corner_heatmap_quality": heatmap_quality,
        "mask_corner_agreement": f["corner_mask_agreement_mean"],
        "mask_corner_joint": mask_corner_joint,
    }


def fit_structured(train, validation, feature_indices, seed):
    best = None
    for regularization in (0.01, 0.1, 1.0, 10.0):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        classifier.fit(train["features"][:, feature_indices], train["labels"])
        scores = classifier.predict_proba(validation["features"][:, feature_indices])[:, 1]
        threshold, balanced = choose_threshold(validation["labels"], scores)
        candidate = (balanced, -metrics(validation, scores, threshold)["negative_false_positive_rate"])
        if best is None or candidate > best[0]:
            best = (candidate, classifier, regularization, threshold, scores)
    return best[1:]


def make_loaders(arguments):
    synthetic = {
        split: MultiTaskDataset(arguments.gate_dataset, arguments.gate_targets, arguments.gate_split_file, split)
        for split in ("train", "validation", "test")
    }
    negative = {
        "train": NoGateDataset(arguments.no_gate_dataset, (0, 1, 2)),
        "validation": NoGateDataset(arguments.no_gate_dataset, (3,)),
        "test": NoGateDataset(arguments.no_gate_dataset, (4,)),
    }
    real = {
        "train": RealGateDataset(arguments.real_root, ("flight_06",)),
        "validation": RealGateDataset(arguments.real_root, ("flight_07",)),
        "test": RealGateDataset(arguments.real_root, ("flight_08",)),
    }

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=arguments.workers,
            pin_memory=True,
            persistent_workers=arguments.workers > 0,
        )

    return {
        split: {domain: loader(collection[split]) for domain, collection in (
            ("synthetic", synthetic), ("negative", negative), ("real", real)
        )}
        for split in ("train", "validation", "test")
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "obstacle-dataset", "gate-dataset", "gate-targets", "gate-split-file",
        "no-gate-dataset", "real-root", "obstacle-checkpoint", "gate-checkpoint",
        "mixed-checkpoint", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = json.loads((arguments.obstacle_dataset / "dataset_manifest.json").read_text())["camera_calibration"]
    model = RetainedObstacleGateModel(
        arguments.obstacle_checkpoint, arguments.gate_checkpoint, camera
    ).to(device)
    checkpoint = torch.load(arguments.mixed_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    print(json.dumps({"device": str(device), "checkpoint_epoch": checkpoint["epoch"]}), flush=True)

    loaders = make_loaders(arguments)
    data = {}
    for split in ("train", "validation", "test"):
        parts = []
        for domain in ("synthetic", "negative", "real"):
            part = collect(model, loaders[split][domain], device, domain)
            parts.append(part)
            print(json.dumps({"split": split, "domain": domain, "examples": len(part["labels"])}), flush=True)
        data[split] = merge(*parts)
        np.savez_compressed(arguments.output / f"{split}_features.npz", **data[split])

    direct = {split: direct_scores(data[split]["features"]) for split in data}
    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold_selection": "maximize balanced accuracy on validation; test is untouched",
        "feature_names": list(FEATURE_NAMES),
        "methods": {},
    }
    for method in direct["validation"]:
        threshold, _ = choose_threshold(data["validation"]["labels"], direct["validation"][method])
        result["methods"][method] = {
            "validation": metrics(data["validation"], direct["validation"][method], threshold),
            "test": metrics(data["test"], direct["test"][method], threshold),
        }

    presence_index = FEATURE_NAMES.index("presence_logit")
    structured = tuple(index for index in range(len(FEATURE_NAMES)) if index != presence_index)
    model_specs = {
        "structured_fusion_no_presence": structured,
        "structured_fusion_with_presence": tuple(range(len(FEATURE_NAMES))),
    }
    for method, indices in model_specs.items():
        classifier, regularization, threshold, validation_scores = fit_structured(
            data["train"], data["validation"], indices, arguments.seed
        )
        test_scores = classifier.predict_proba(data["test"]["features"][:, indices])[:, 1]
        result["methods"][method] = {
            "regularization_C": regularization,
            "features": [FEATURE_NAMES[index] for index in indices],
            "validation": metrics(data["validation"], validation_scores, threshold),
            "test": metrics(data["test"], test_scores, threshold),
        }

    ranking = sorted(
        result["methods"],
        key=lambda method: (
            result["methods"][method]["test"]["balanced_accuracy"],
            result["methods"][method]["test"]["auroc"],
        ),
        reverse=True,
    )
    result["held_out_ranking"] = ranking
    (arguments.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    with (arguments.output / "comparison.csv").open("w", newline="") as stream:
        fields = ["method", "split", "auroc", "average_precision", "balanced_accuracy", "accuracy",
                  "negative_false_positive_rate", "synthetic_recall", "real_recall", "bce", "brier", "threshold"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, values in result["methods"].items():
            for split in ("validation", "test"):
                row = {"method": method, "split": split}
                row.update({key: values[split].get(key) for key in fields[2:]})
                writer.writerow(row)
    print(json.dumps({"held_out_ranking": ranking, "output": str(arguments.output)}), flush=True)


if __name__ == "__main__":
    main()
