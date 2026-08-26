#!/usr/bin/env python3
"""Ground-truth evaluation of the four exported NeMO integer ONNX graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_REPO = Path("/home/cchen/tinympc-gate-texture")
ISAAC_REPO = Path("/home/cchen/isaacsim-workspace")
sys.path[:0] = [str(ISAAC_REPO), str(TRAINING_REPO)]

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

import gap8_perception as _gap8_package
_gap8_package.__path__.append(str(TRAINING_REPO / "gap8_perception"))
from gap8_perception.data import MultiTaskDataset  # noqa: E402
from gap8_perception.evaluate import local_centroid  # noqa: E402
from gap8_perception.temporal_data import TemporalHorizonDataset  # noqa: E402
from gap8_perception.train_espnet_dory_student import (  # noqa: E402
    auc,
    average_precision,
)
from user_workflows.train_retained_obstacle_gate import (  # noqa: E402
    NoGateDataset,
    RealGateDataset,
    RetainedObstacleGateModel,
)


class IntegerGraphs:
    def __init__(self, root: Path, report_path: Path):
        report = json.loads(report_path.read_text())
        self.reports = {item["graph"]: item for item in report["graphs"]}
        self.sessions = {
            name: ort.InferenceSession(
                str(root / name / (name + "_int.onnx")),
                providers=["CPUExecutionProvider"],
            )
            for name in ("encoder", "corner_head", "gate_head", "danger_head")
        }

    def run(self, frames: np.ndarray):
        frames = np.asarray(frames, np.float32)[None] * 255.0
        encoder = self.sessions["encoder"]
        shared = encoder.run(None, {encoder.get_inputs()[0].name: frames})[0]
        decoded = {}
        for name in ("corner_head", "gate_head", "danger_head"):
            session = self.sessions[name]
            raw = session.run(None, {session.get_inputs()[0].name: shared})[0][0]
            graph = self.reports[name]
            epsilon = float(graph["output_epsilon"])
            offset = np.asarray(graph["output_offset"], np.float32)[:, None, None]
            bias = np.asarray(graph["learned_bias"], np.float32)[:, None, None]
            decoded[name] = raw * epsilon - offset + bias
        return decoded


def evenly_spaced(length: int, maximum: int):
    if maximum <= 0 or length <= maximum:
        return range(length)
    return np.unique(np.linspace(0, length - 1, maximum).astype(int)).tolist()


def obstacle_raw(graphs, dataset, teacher, maximum):
    labels, scores = [], []
    intersection = union = 0
    for index in evenly_spaced(len(dataset), maximum):
        sample = dataset[index]
        logits = graphs.run(sample["images"].numpy())["danger_head"]
        probability = torch.from_numpy(logits).sigmoid()
        inverse_depth = sample["inverse_depth"].clamp(0.0, 1.0)
        valid = sample["depth_valid"].to(inverse_depth.dtype)
        near = F.adaptive_max_pool2d((inverse_depth * valid)[None], (10, 10))[0]
        corridor = teacher.obstacle.projector(
            sample["horizon"][None], sample["horizon_mask"][None], 10, 10
        ).amax(1)[0].cpu()
        intersection += int(((probability >= 0.5) & (near >= 0.5)).sum())
        union += int(((probability >= 0.5) | (near >= 0.5)).sum())
        scores.append(float((probability * corridor).amax()))
        labels.append(bool(sample["collision"]))
    return np.asarray(labels, bool), np.asarray(scores), intersection, union


def obstacle_report(raw, threshold):
    labels, scores, intersection, union = raw
    prediction = scores >= threshold
    positives = labels.sum()
    negatives = (~labels).sum()
    return {
        "examples": int(len(labels)),
        "collision_positive_count": int(positives),
        "danger_iou_at_0_5": intersection / max(1, union),
        "collision_auroc": auc(labels, scores),
        "collision_ap": average_precision(labels, scores),
        "collision_threshold": float(threshold),
        "collision_recall": float(prediction[labels].mean()) if positives else None,
        "collision_false_positive_rate": (
            float(prediction[~labels].mean()) if negatives else None
        ),
    }


def select_threshold(raw, minimum_recall):
    labels, scores = raw[:2]
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    feasible = [
        threshold for threshold in candidates
        if (scores[labels] >= threshold).mean() >= minimum_recall
    ]
    if not feasible:
        raise RuntimeError("integer danger output cannot meet recall constraint")
    return float(max(feasible))


def gate_report(graphs, synthetic, negative, real, maximum):
    intersection = union = negative_pixels = negative_total = 0
    errors, real_errors = [], []
    for index in evenly_spaced(len(synthetic), maximum):
        sample = synthetic[index]
        image = sample["image"].numpy()
        output = graphs.run(np.repeat(image, 2, axis=0))
        prediction = torch.from_numpy(output["gate_head"]).sigmoid() >= 0.5
        target = sample["gate"] >= 0.5
        intersection += int((prediction & target).sum())
        union += int((prediction | target).sum())
        if bool(sample["corner_valid"]):
            points = local_centroid(
                torch.from_numpy(output["corner_head"])[None].sigmoid()
            )[0] * 4.0
            errors.extend(torch.linalg.vector_norm(
                points - sample["corner_xy"], dim=1
            ).tolist())
    for index in evenly_spaced(len(negative), maximum):
        sample = negative[index]
        image = sample["image"].numpy()
        output = graphs.run(np.repeat(image, 2, axis=0))
        prediction = torch.from_numpy(output["gate_head"]).sigmoid() >= 0.5
        negative_pixels += int(prediction.sum())
        negative_total += prediction.numel()
    for index in evenly_spaced(len(real), maximum):
        sample = real[index]
        image = sample["image"].numpy()
        output = graphs.run(np.repeat(image, 2, axis=0))
        points = local_centroid(
            torch.from_numpy(output["corner_head"])[None].sigmoid()
        )[0] * 4.0
        real_errors.extend(torch.linalg.vector_norm(
            points - sample["corner_xy"], dim=1
        ).tolist())
    errors = np.asarray(errors)
    real_errors = np.asarray(real_errors)
    return {
        "synthetic_examples": len(evenly_spaced(len(synthetic), maximum)),
        "negative_examples": len(evenly_spaced(len(negative), maximum)),
        "real_examples": len(evenly_spaced(len(real), maximum)),
        "gate_iou": intersection / max(1, union),
        "corner_mean_px": float(errors.mean()),
        "corner_p95_px": float(np.percentile(errors, 95)),
        "real_corner_mean_px": float(real_errors.mean()),
        "real_corner_p95_px": float(np.percentile(real_errors, 95)),
        "negative_mask_pixel_rate": negative_pixels / max(1, negative_total),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "integer-dir", "nemo-report", "obstacle-dataset", "gate-dataset",
        "gate-targets", "gate-split-file", "no-gate-dataset",
        "paired-no-gate-dataset", "real-root", "obstacle-checkpoint",
        "gate-checkpoint", "teacher-checkpoint", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maximum-per-dataset", type=int, default=0)
    parser.add_argument("--minimum-collision-recall", type=float, default=0.95)
    args = parser.parse_args()
    graphs = IntegerGraphs(args.integer_dir, args.nemo_report)
    camera = json.loads(
        (args.obstacle_dataset / "dataset_manifest.json").read_text()
    )["camera_calibration"]
    teacher = RetainedObstacleGateModel(
        args.obstacle_checkpoint, args.gate_checkpoint, camera
    ).cpu().eval()
    teacher.load_state_dict(torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )["model"])
    obstacle_validation = TemporalHorizonDataset(
        args.obstacle_dataset, "validation", 2, minimum_current_index=2
    )
    obstacle_test = TemporalHorizonDataset(
        args.obstacle_dataset, "test", 2, minimum_current_index=2
    )
    validation_raw = obstacle_raw(
        graphs, obstacle_validation, teacher, args.maximum_per_dataset
    )
    threshold = select_threshold(validation_raw, args.minimum_collision_recall)
    test_raw = obstacle_raw(graphs, obstacle_test, teacher, args.maximum_per_dataset)
    synthetic = MultiTaskDataset(
        args.gate_dataset, args.gate_targets, args.gate_split_file, "test"
    )
    negative = ConcatDataset((
        NoGateDataset(args.no_gate_dataset, (4,)),
        NoGateDataset(args.paired_no_gate_dataset, range(18, 20)),
    ))
    real = RealGateDataset(args.real_root, ("flight_08",))
    report = {
        "format": "espnet-dory-student-integer-ground-truth-v1",
        "threshold_selected_on": "validation",
        "selected_danger_threshold": threshold,
        "obstacle_validation": obstacle_report(validation_raw, threshold),
        "obstacle_test": obstacle_report(test_raw, threshold),
        "gate_test": gate_report(
            graphs, synthetic, negative, real, args.maximum_per_dataset
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
