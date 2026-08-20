#!/usr/bin/env python3
"""Evaluate the five NeMO integer graphs on held-out navigation and gate data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset

TRAINING_ROOT = Path("/home/cchen/tinympc-gate-texture")
PULP_ROOT = Path("/home/cchen/pulp-dronet/tiny-pulp-dronet-v3")
sys.path[:0] = [str(TRAINING_ROOT), str(PULP_ROOT)]

from gap8_perception.audit_real_flights import canonical_image_order  # noqa: E402
from gap8_perception.data import MultiTaskDataset  # noqa: E402
from gap8_perception.evaluate import local_centroid  # noqa: E402


@dataclass(frozen=True)
class PairSample:
    previous: Path
    current: Path
    yaw_rate: float
    collision: float


def find_strict_pairs(root: Path, partition: str):
    samples = []
    for labels_path in sorted(root.rglob("labels_partitioned.csv")):
        with labels_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        images_dir = labels_path.parent / "images"
        disk_files = sorted(
            (path for path in images_dir.glob("*.jpeg") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        rank = {path.name: index for index, path in enumerate(disk_files)}
        for previous, current in zip(rows[:-1], rows[1:]):
            if previous["partition"] != partition or current["partition"] != partition:
                continue
            p_name, c_name = previous["filename"], current["filename"]
            if p_name not in rank or c_name not in rank or rank[c_name] != rank[p_name] + 1:
                continue
            samples.append(PairSample(
                images_dir / p_name, images_dir / c_name,
                float(current["label_yaw_rate"]) / 90.0,
                float(current["label_collision"]),
            ))
    return samples


class MatchedFrames(Dataset):
    def __init__(self, samples, _frame_count=2, _augment=False):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for path in (sample.previous, sample.current):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            top, left = (height - 200) // 2, (width - 200) // 2
            images.append(image[top:top + 200, left:left + 200])
        return (
            torch.from_numpy(np.stack(images)).float() / 255.0,
            torch.tensor((sample.yaw_rate, sample.collision), dtype=torch.float32),
        )


class NoGateDataset(Dataset):
    def __init__(self, root: Path, shard_indices):
        self.paths = []
        for index in shard_indices:
            shard = root / ("shard_%09d" % (index * 1000))
            self.paths.extend(sorted(shard.glob("hm01b0_mono_*.png")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        return {"image": torch.from_numpy(image).unsqueeze(0).float() / 255.0}


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
                self.records.append((folder / "stream_out" / row["image"], corners))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, corners = self.records[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
            "corner_xy": torch.from_numpy(corners.copy()),
        }


def binary_auroc(truth, scores):
    positives, negatives = int(truth.sum()), len(truth) - int(truth.sum())
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth, sorted_scores = truth[order], scores[order]
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], len(scores) - 1]
    tp = np.cumsum(sorted_truth)[distinct]
    fp = 1 + distinct - tp
    return float(np.trapezoid(np.r_[0.0, tp / positives], np.r_[0.0, fp / negatives]))


def average_precision(truth, scores):
    positives = int(truth.sum())
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth, sorted_scores = truth[order], scores[order]
    distinct = np.r_[np.where(np.diff(sorted_scores))[0], len(scores) - 1]
    tp = np.cumsum(sorted_truth)[distinct]
    recall, precision = tp / positives, tp / (distinct + 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def threshold_metrics(truth, scores, threshold):
    prediction, truth = scores >= threshold, truth.astype(bool)
    tp, tn = int((prediction & truth).sum()), int((~prediction & ~truth).sum())
    fp, fn = int((prediction & ~truth).sum()), int((~prediction & truth).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "threshold": float(threshold), "accuracy": (tp + tn) / len(truth),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


class IntegerGraphs:
    names = ("encoder", "corner_head", "gate_head", "presence_head", "navigation_head")

    def __init__(self, root: Path, report_path: Path):
        report = json.loads(report_path.read_text())
        self.reports = {item["graph"]: item for item in report["graphs"]}
        self.logit_scales = report.get("deployment_logit_scale", {})
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.sessions = {
            name: ort.InferenceSession(
                str(root / name / (name + "_int.onnx")),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name in self.names
        }

    def _session(self, name: str, value: np.ndarray) -> np.ndarray:
        session = self.sessions[name]
        return session.run(None, {session.get_inputs()[0].name: value})[0]

    def run(self, frames: torch.Tensor, names=None) -> dict[str, np.ndarray]:
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        frames = F.interpolate(frames.float(), (160, 160), mode="bilinear", align_corners=False)
        shared = self._session("encoder", frames.numpy() * 255.0)
        decoded = {}
        for name in (names or self.names[1:]):
            raw = self._session(name, shared)
            graph = self.reports[name]
            channels = raw.shape[1]
            shape = (1, channels) + (1,) * (raw.ndim - 2)
            epsilon = float(graph["output_epsilon"])
            offset = np.asarray(graph["output_offset"], np.float32).reshape(shape)
            bias = np.asarray(graph["learned_bias"], np.float32).reshape(shape)
            decoded[name] = raw * epsilon - offset + bias
            scale = np.asarray(
                self.logit_scales.get(name, [1.0] * channels), np.float32
            ).reshape(shape)
            decoded[name] *= scale
        return decoded


def spaced(length: int, maximum: int):
    if maximum <= 0 or length <= maximum:
        return range(length)
    return np.unique(np.linspace(0, length - 1, maximum).astype(int)).tolist()


def navigation_raw(graphs: IntegerGraphs, dataset, maximum: int):
    truth, scores, yaws, yaw_truth = [], [], [], []
    for index in spaced(len(dataset), maximum):
        frames, labels = dataset[index]
        output = graphs.run(frames, ("navigation_head",))["navigation_head"].reshape(-1)
        yaws.append(float(output[0]))
        scores.append(float(torch.sigmoid(torch.tensor(output[1]))))
        yaw_truth.append(float(labels[0]))
        truth.append(int(labels[1]))
    return tuple(np.asarray(value) for value in (truth, scores, yaw_truth, yaws))


def best_f1_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    return float(max(candidates, key=lambda value: threshold_metrics(truth, scores, value)["f1"]))


def navigation_report(raw, threshold: float) -> dict:
    truth, scores, yaw_truth, yaws = raw
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    mse = float(np.mean(np.square(yaws - yaw_truth)))
    bce = float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))
    return {
        "samples": int(len(truth)),
        "yaw_rmse": math.sqrt(mse),
        "collision_bce": bce,
        "collision_auroc": binary_auroc(truth, scores),
        "collision_ap": average_precision(truth, scores),
        "threshold_metrics": threshold_metrics(truth, scores, threshold),
    }


def fusion_score(features: np.ndarray, fusion: dict) -> np.ndarray:
    standardized = (features - np.asarray(fusion["mean"])) / np.asarray(fusion["scale"])
    logits = standardized @ np.asarray(fusion["weight"]) + float(fusion["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def gate_report(graphs, synthetic, negative, real, fusion, maximum):
    intersection = union = negative_pixels = negative_total = 0
    errors, real_errors, labels, features = [], [], [], []

    def infer(image):
        output = graphs.run(
            image.repeat(2, 1, 1),
            ("corner_head", "gate_head", "presence_head"),
        )
        corner_logits = torch.from_numpy(output["corner_head"])
        gate_logits = torch.from_numpy(output["gate_head"])
        corners = F.interpolate(corner_logits, (40, 40), mode="bilinear", align_corners=False)
        gate = F.interpolate(gate_logits, (40, 40), mode="bilinear", align_corners=False)
        presence = float(torch.sigmoid(torch.from_numpy(output["presence_head"])).item())
        feature = (
            presence,
            float(gate.sigmoid().flatten().topk(16).values.mean()),
            float(corners.sigmoid().flatten(2).amax(2).mean()),
        )
        return corners, gate, feature

    for index in spaced(len(synthetic), maximum):
        sample = synthetic[index]
        corners, gate, feature = infer(sample["image"])
        prediction, target = gate.sigmoid()[0] >= 0.5, sample["gate"] >= 0.5
        intersection += int((prediction & target).sum())
        union += int((prediction | target).sum())
        if bool(sample["corner_valid"]):
            points = local_centroid(corners.sigmoid())[0] * 4.0
            errors.extend(torch.linalg.vector_norm(points - sample["corner_xy"], dim=1).tolist())
        labels.append(1)
        features.append(feature)
    for index in spaced(len(negative), maximum):
        sample = negative[index]
        corners, gate, feature = infer(sample["image"])
        prediction = gate.sigmoid() >= 0.5
        negative_pixels += int(prediction.sum())
        negative_total += prediction.numel()
        labels.append(0)
        features.append(feature)
    for index in spaced(len(real), maximum):
        sample = real[index]
        corners, _, feature = infer(sample["image"])
        points = local_centroid(corners.sigmoid())[0] * 4.0
        real_errors.extend(torch.linalg.vector_norm(points - sample["corner_xy"], dim=1).tolist())
        labels.append(1)
        features.append(feature)
    labels = np.asarray(labels)
    confidence = fusion_score(np.asarray(features), fusion)
    errors, real_errors = np.asarray(errors), np.asarray(real_errors)
    return {
        "synthetic_examples": len(spaced(len(synthetic), maximum)),
        "negative_examples": len(spaced(len(negative), maximum)),
        "real_examples": len(spaced(len(real), maximum)),
        "gate_iou": intersection / max(1, union),
        "corner_mean_px": float(errors.mean()),
        "corner_p95_px": float(np.percentile(errors, 95)),
        "real_corner_mean_px": float(real_errors.mean()),
        "real_corner_p95_px": float(np.percentile(real_errors, 95)),
        "negative_mask_pixel_rate": negative_pixels / max(1, negative_total),
        "structured_confidence": {
            "auroc": binary_auroc(labels, confidence),
            "ap": average_precision(labels, confidence),
            **threshold_metrics(labels, confidence, float(fusion["threshold"])),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "integer-dir", "nemo-report", "training-summary", "navigation-data",
        "gate-dataset", "gate-targets", "gate-split-file", "no-gate-dataset",
        "paired-no-gate-dataset", "real-root", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maximum-per-dataset", type=int, default=0)
    args = parser.parse_args()
    graphs = IntegerGraphs(args.integer_dir, args.nemo_report)
    validation = MatchedFrames(find_strict_pairs(args.navigation_data, "valid"), 2, False)
    test = MatchedFrames(find_strict_pairs(args.navigation_data, "test"), 2, False)
    validation_raw = navigation_raw(graphs, validation, args.maximum_per_dataset)
    threshold = best_f1_threshold(validation_raw[0], validation_raw[1])
    synthetic = MultiTaskDataset(
        args.gate_dataset, args.gate_targets, args.gate_split_file, "test"
    )
    negative = ConcatDataset((
        NoGateDataset(args.no_gate_dataset, (4,)),
        NoGateDataset(args.paired_no_gate_dataset, range(18, 20)),
    ))
    real = RealGateDataset(args.real_root, ("flight_08",))
    training = json.loads(args.training_summary.read_text())
    report = {
        "format": "espnet-dronet-middle-gate-integer-task-evaluation-v1",
        "input_resolution": [160, 160],
        "navigation_threshold_selected_on": "official validation split",
        "navigation_validation": navigation_report(validation_raw, threshold),
        "navigation_test": navigation_report(
            navigation_raw(graphs, test, args.maximum_per_dataset), threshold
        ),
        "gate_test": gate_report(
            graphs, synthetic, negative, real,
            training["structured_confidence_fusion"], args.maximum_per_dataset,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
