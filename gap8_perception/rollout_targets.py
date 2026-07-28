"""Geometric range targets and exact state-conditioned collision rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def _box(center, size):
    center, size = np.asarray(center, np.float32), np.asarray(size, np.float32)
    return np.stack((center - size / 2, center + size / 2))


def course_collision_boxes() -> np.ndarray:
    """AABBs matching the fixed generator scene's collision geometry."""
    boxes = [
        _box((0, 0, -0.08), (8, 8, 0.12)),
        _box((0, 2.0, 0.04), (4.1, 0.06, 0.08)),
        _box((0, -2.0, 0.04), (4.1, 0.06, 0.08)),
        _box((-2.0, 0, 0.04), (0.06, 4.1, 0.08)),
        _box((2.0, 0, 0.04), (0.06, 4.1, 0.08)),
        _box((-0.55, 0.65, 0.30), (0.55, 0.55, 0.60)),
        _box((0.60, -0.65, 0.38), (0.50, 0.75, 0.76)),
    ]
    gate_outer, gate_inner = 0.66, 0.45
    frame = (gate_outer - gate_inner) / 2
    offset, center_z = gate_inner / 2 + frame / 2, 0.55
    for x, y in ((-1.30, -0.45), (1.30, 0.45)):
        boxes.extend(
            [
                _box((x, y - offset, center_z), (0.025, frame, gate_outer)),
                _box((x, y + offset, center_z), (0.025, frame, gate_outer)),
                _box((x, y, center_z - offset), (0.025, gate_outer, frame)),
                _box((x, y, center_z + offset), (0.025, gate_outer, frame)),
            ]
        )
    for center, size in [
        ((-2.75, 0.0, 0.78), (1.1, 2.0, 0.10)),
        ((2.75, 0.1, 0.78), (1.1, 1.8, 0.10)),
        ((-2.70, 0.0, 1.15), (0.12, 0.65, 0.48)),
        ((2.70, 0.1, 1.15), (0.12, 0.65, 0.48)),
        ((0.0, 3.10, 1.10), (3.0, 0.45, 2.2)),
        ((-3.25, -2.7, 1.00), (0.60, 1.5, 2.0)),
    ]:
        boxes.append(_box(center, size))
    return np.asarray(boxes, dtype=np.float32)


def camera_rays_world(
    eye,
    target,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    output_size: int = 20,
) -> np.ndarray:
    centers = (
        np.stack(
            np.meshgrid(
                (np.arange(output_size) + 0.5) * (160 / output_size),
                (np.arange(output_size) + 0.5) * (160 / output_size),
            ),
            axis=-1,
        )
        .reshape(-1, 1, 2)
        .astype(np.float64)
    )
    normalized = cv2.undistortPoints(
        centers, camera_matrix.astype(np.float64), distortion.astype(np.float64)
    ).reshape(-1, 2)
    eye, target = np.asarray(eye, np.float32), np.asarray(target, np.float32)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rays = (
        normalized[:, 0:1] * right
        - normalized[:, 1:2] * up
        + forward
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays.astype(np.float32)


def signed_clearance_to_boxes(
    points: np.ndarray, boxes: np.ndarray, clearance_radius: float
) -> np.ndarray:
    """Minimum signed sphere-to-AABB clearance for N points."""
    point = points[:, None, :]
    low, high = boxes[None, :, 0], boxes[None, :, 1]
    outside_delta = np.maximum(np.maximum(low - point, point - high), 0)
    outside = np.linalg.norm(outside_delta, axis=2)
    inside = ((point >= low) & (point <= high)).all(axis=2)
    face_distance = np.minimum(point - low, high - point).min(axis=2)
    signed = np.where(inside, -face_distance, outside) - clearance_radius
    return signed.min(axis=1)


def conservative_range_to_boxes(
    eye: np.ndarray,
    rays: np.ndarray,
    boxes: np.ndarray,
    clearance_radius: float,
    max_range_m: float = 6.0,
) -> np.ndarray:
    """First ray intersection with sphere-inflated AABBs, clipped to max range."""
    origin = np.asarray(eye, np.float32)[None, None, :]
    direction = np.asarray(rays, np.float32)[:, None, :]
    low = boxes[None, :, 0] - clearance_radius
    high = boxes[None, :, 1] + clearance_radius
    safe_direction = np.where(
        np.abs(direction) < 1e-8,
        np.where(direction < 0, -1e-8, 1e-8),
        direction,
    )
    t0 = (low - origin) / safe_direction
    t1 = (high - origin) / safe_direction
    entry = np.maximum(np.minimum(t0, t1), 0.0).max(axis=2)
    exit_ = np.maximum(t0, t1).min(axis=2)
    hit = exit_ >= entry
    distances = np.where(hit, entry, np.inf).min(axis=1)
    return np.minimum(distances, max_range_m).astype(np.float32)


def simulate_candidate_rollouts(
    eye,
    target,
    camera_matrix,
    distortion,
    speed_mps: float,
    horizon_s: float = 1.0,
    timestep_s: float = 0.05,
    latency_s: float = 0.08,
    drone_radius_m: float = 0.10,
    safety_margin_m: float = 0.10,
    acceleration_limit_mps2: float = 6.0,
    attitude_limit_deg: float = 35.0,
    boxes: np.ndarray | None = None,
):
    boxes = course_collision_boxes() if boxes is None else boxes
    rays = camera_rays_world(eye, target, camera_matrix, distortion)
    eye = np.asarray(eye, np.float32)
    forward = np.asarray(target, np.float32) - eye
    forward /= np.linalg.norm(forward)
    current_velocity = forward * speed_mps
    position = np.repeat(
        (eye + current_velocity * latency_s)[None], len(rays), axis=0
    )
    velocity = np.repeat(current_velocity[None], len(rays), axis=0)
    desired_velocity = rays * speed_mps
    max_acceleration = min(
        acceleration_limit_mps2,
        9.81 * math.tan(math.radians(attitude_limit_deg)),
    )
    steps = int(math.ceil(horizon_s / timestep_s))
    minimum_clearance = signed_clearance_to_boxes(
        position, boxes, drone_radius_m + safety_margin_m
    )
    collision = minimum_clearance <= 0
    ttc = np.where(collision, latency_s, -1.0).astype(np.float32)
    for step in range(1, steps + 1):
        delta_v = desired_velocity - velocity
        norm = np.linalg.norm(delta_v, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_acceleration * timestep_s / (norm + 1e-8))
        velocity += delta_v * scale
        position += velocity * timestep_s
        clearance = signed_clearance_to_boxes(
            position, boxes, drone_radius_m + safety_margin_m
        )
        minimum_clearance = np.minimum(minimum_clearance, clearance)
        newly_colliding = (~collision) & (clearance <= 0)
        ttc[newly_colliding] = latency_s + step * timestep_s
        collision |= newly_colliding
    total_time = latency_s + horizon_s
    urgency = np.where(collision, 1.0 - np.clip(ttc / total_time, 0, 1), 0.0)
    uncertainty = np.exp(-np.abs(minimum_clearance) / 0.15)
    range_m = conservative_range_to_boxes(
        eye, rays, boxes, drone_radius_m + safety_margin_m
    )
    inverse_range = 1.0 - np.clip(range_m / 6.0, 0.0, 1.0)
    return {
        "collision": collision.reshape(20, 20).astype(np.uint8),
        "minimum_clearance_m": minimum_clearance.reshape(20, 20).astype(np.float32),
        "ttc_s": ttc.reshape(20, 20).astype(np.float32),
        "urgency": urgency.reshape(20, 20).astype(np.float32),
        "uncertainty": uncertainty.reshape(20, 20).astype(np.float32),
        "range_m": range_m.reshape(20, 20).astype(np.float32),
        "inverse_range": inverse_range.reshape(20, 20).astype(np.float32),
    }


def load_calibration(path: Path):
    calibration = json.loads(path.read_text())
    return (
        np.asarray(calibration["camera_matrix"], np.float32),
        np.asarray(calibration["distortion_coefficients"], np.float32),
    )
