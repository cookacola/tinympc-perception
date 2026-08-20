#!/usr/bin/env python3
"""Audit noisy real-flight gate labels without treating them as exact truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def nearest_edge_distance(image: np.ndarray, point: np.ndarray, radius: int = 7):
    x, y = np.rint(point).astype(int)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    edges = cv2.Canny(image, 40, 100)
    candidates = np.argwhere(edges[y0:y1, x0:x1] > 0)
    if len(candidates):
        candidates_xy = candidates[:, ::-1].astype(np.float32)
        candidates_xy += np.array([x0, y0], np.float32)
        snapped = candidates_xy[
            np.linalg.norm(candidates_xy - point, axis=1).argmin()
        ]
    else:
        snapped = point.copy()
    return float(np.linalg.norm(snapped - point)), snapped


def canonical_image_order(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return TL,TR,BR,BL based on the image-plane top and bottom pairs."""
    top = np.argsort(points[:, 1])[:2]
    bottom = np.argsort(points[:, 1])[2:]
    tl, tr = top[np.argsort(points[top, 0])]
    bl, br = bottom[np.argsort(points[bottom, 0])]
    permutation = (int(tl), int(tr), int(br), int(bl))
    return points[list(permutation)], permutation


def topology_valid(points: np.ndarray) -> bool:
    canonical = canonical_image_order(points)[0]
    tl, tr, br, bl = canonical
    signed_area = 0.5 * np.sum(
        canonical[:, 0] * np.roll(canonical[:, 1], -1)
        - canonical[:, 1] * np.roll(canonical[:, 0], -1)
    )
    return bool(
        tl[1] < bl[1]
        and tr[1] < br[1]
        and tl[0] < tr[0]
        and bl[0] < br[0]
        and signed_area > 0
    )


def load_image_times(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        return {row["image"]: float(row["timestamp"]) for row in csv.DictReader(stream)}


def combined_order_audit(root: Path, authoritative: dict[tuple[str, str], np.ndarray]):
    permutation_counts = Counter()
    compared = 0
    for manifest in (root / "dataset").glob("*.jsonl"):
        for line in manifest.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("flight", ""), Path(row["image"]).name)
            if key not in authoritative or not row.get("corners"):
                continue
            candidate = np.asarray(row["corners"], np.float32).reshape(4, 2)
            _, permutation = canonical_image_order(candidate)
            permutation_counts[str(permutation)] += 1
            compared += 1
    return {
        "records_compared": compared,
        "raw_to_image_TL_TR_BR_BL_permutation_counts": dict(permutation_counts),
        "interpretation": (
            "raw combined and per-flight sequences are not reliable semantic "
            "TL,TR,BR,BL labels; canonicalize in image space before use"
        ),
    }


def mocap_speed_summary(samples: np.ndarray) -> dict:
    """Summarize translational speed while rejecting timestamp discontinuities."""
    delta_t = np.diff(samples[:, 0])
    valid = (delta_t >= 0.001) & (delta_t <= 0.2)
    distance = np.linalg.norm(np.diff(samples[:, 1:4], axis=0), axis=1)
    speed = distance[valid] / delta_t[valid]
    return {
        "samples": int(speed.size),
        "median_m_per_s": float(np.median(speed)),
        "p95_m_per_s": float(np.percentile(speed, 95)),
        "p99_m_per_s": float(np.percentile(speed, 99)),
        "maximum_m_per_s": float(np.max(speed)),
        "fraction_at_or_above_1_m_per_s": float(np.mean(speed >= 1.0)),
        "fraction_at_or_above_2_m_per_s": float(np.mean(speed >= 2.0)),
        "interpretation": (
            "finite-difference mocap diagnostic, not proof of closed-loop "
            "model operation or sustained racing speed"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/cchen/real_flight_data")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    report = {
        "root": str(args.root),
        "dataset_scope": {
            "scene_content": "gates_only_no_obstacles",
            "valid_tasks": ["gate_corner_detection", "gate_geometry_verification"],
            "excluded_tasks": [
                "obstacle_presence",
                "inverse_range",
                "collision_danger",
                "collision_urgency",
            ],
            "unlabeled_frame_policy": (
                "unlabeled means no usable gate-corner annotation; it must not "
                "be converted to an obstacle, range, or danger negative"
            ),
        },
        "flights": {},
    }
    authoritative = {}
    sample_pool = []
    all_edge_distances = []
    all_sync_ms = []

    for flight in ("flight_06", "flight_07", "flight_08"):
        folder = args.root / flight
        labels = [
            json.loads(line)
            for line in (folder / "labels.jsonl").read_text().splitlines()
            if line
        ]
        image_times = load_image_times(folder / "stream_out" / "image_times.csv")
        with (folder / f"{flight}.csv").open(newline="") as stream:
            mocap = np.asarray(
                [
                    [
                        float(row["timestamp"]),
                        float(row["drone_px"]),
                        float(row["drone_py"]),
                        float(row["drone_pz"]),
                    ]
                    for row in csv.DictReader(stream)
                ],
                np.float64,
            )
        mocap_times = mocap[:, 0]
        image_count = len(list((folder / "stream_out").glob("img_*.png")))
        topology_ok = 0
        raw_to_canonical = Counter()
        edge_distances = []
        sync_ms = []
        for row in labels:
            image_path = folder / "stream_out" / row["image"]
            points = np.asarray(row["corners"], np.float32).reshape(4, 2)
            authoritative[(flight, row["image"])] = points
            topology_ok += int(topology_valid(points))
            canonical, permutation = canonical_image_order(points)
            raw_to_canonical[str(permutation)] += 1
            timestamp = image_times[row["image"]]
            position = int(np.searchsorted(mocap_times, timestamp))
            neighbors = mocap_times[max(0, position - 1) : min(len(mocap_times), position + 1)]
            sync_ms.append(float(np.min(np.abs(neighbors - timestamp)) * 1000.0))
            sample_pool.append((flight, image_path, points))
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            for point in points:
                distance, _ = nearest_edge_distance(image, point)
                edge_distances.append(distance)
        all_edge_distances.extend(edge_distances)
        all_sync_ms.extend(sync_ms)
        report["flights"][flight] = {
            "images": image_count,
            "labeled_positive_frames": len(labels),
            "unlabeled_frames": image_count - len(labels),
            "negative_labeled_frames": 0,
            "recorded_mocap_speed": mocap_speed_summary(mocap),
            "corner_topology_valid_fraction": topology_ok / len(labels),
            "raw_to_image_TL_TR_BR_BL_permutation_counts": dict(raw_to_canonical),
            "nearest_mocap_timestamp_ms": {
                "median": float(np.median(sync_ms)),
                "p95": float(np.percentile(sync_ms, 95)),
                "maximum": float(np.max(sync_ms)),
            },
            "label_to_local_strong_edge_px": {
                "median": float(np.median(edge_distances)),
                "p95": float(np.percentile(edge_distances, 95)),
                "maximum": float(np.max(edge_distances)),
                "note": "diagnostic only; nearest Canny edge is not corrected ground truth",
            },
        }

    report["combined_manifest_order"] = combined_order_audit(
        args.root, authoritative
    )
    report["aggregate"] = {
        "images": sum(item["images"] for item in report["flights"].values()),
        "labeled_positive_frames": len(sample_pool),
        "label_to_local_strong_edge_px_median": float(np.median(all_edge_distances)),
        "label_to_local_strong_edge_px_p95": float(np.percentile(all_edge_distances, 95)),
        "nearest_mocap_timestamp_ms_p95": float(np.percentile(all_sync_ms, 95)),
    }
    report["recommended_use"] = {
        "primary": "held-out sim-to-real verification grouped by whole flight",
        "corner_labels": "weak/noisy labels; report tolerance bands, do not use as exact heatmap centers",
        "fine_tuning": "only after manual or robust geometric refinement; never split adjacent frames",
        "danger_labels": "not present; do not infer collision-free labels from successful flight alone",
    }

    selected = rng.choice(len(sample_pool), size=min(args.samples, len(sample_pool)), replace=False)
    tiles = []
    colors = ((255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255))
    for index in selected:
        flight, image_path, raw_points = sample_pool[int(index)]
        points, _ = canonical_image_order(raw_points)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(tile, [np.rint(points).astype(np.int32)], True, (0, 0, 255), 1)
        for point, color in zip(points, colors):
            distance, snapped = nearest_edge_distance(image, point)
            cv2.circle(tile, tuple(np.rint(point).astype(int)), 2, color, -1)
            cv2.line(
                tile,
                tuple(np.rint(point).astype(int)),
                tuple(np.rint(snapped).astype(int)),
                (255, 255, 255),
                1,
            )
        cv2.putText(
            tile, f"{flight}:{image_path.stem}", (2, 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1, cv2.LINE_AA,
        )
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 10):
        row = tiles[start : start + 10]
        row.extend([np.zeros_like(tiles[0])] * (10 - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(args.output / "real_flight_100_label_audit.png"), np.vstack(rows))
    (args.output / "real_flight_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
