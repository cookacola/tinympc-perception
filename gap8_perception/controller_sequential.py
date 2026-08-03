"""Reference decoder and TinyMPC half-space construction for the design contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .output_contract import (
    NORMAL_ANGLES_DEG,
    OFFSET_MAX,
    OFFSET_MIN,
    SCORE_LIMIT,
)


@dataclass(frozen=True)
class DecodedSequentialOutput:
    corners_xy_crop: np.ndarray
    corner_peak_scores: np.ndarray
    corner_ambiguity_margins: np.ndarray
    offsets_m: np.ndarray
    confidence_scores: np.ndarray


@dataclass(frozen=True)
class WorldHalfSpaces:
    normals_world: np.ndarray
    bounds_world: np.ndarray
    accepted: np.ndarray
    effective_offsets_m: np.ndarray


@dataclass(frozen=True)
class GatePose:
    rotation_vector: np.ndarray
    translation_vector_m: np.ndarray
    reprojection_error_px: float


def fixed_normals_body() -> np.ndarray:
    angles = np.radians(np.asarray(NORMAL_ANGLES_DEG, np.float64))
    return np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)


def decode_output(output_chw: np.ndarray) -> DecodedSequentialOutput:
    output = np.asarray(output_chw, np.float32)
    if output.shape != (12, 15, 20):
        raise ValueError(f"expected (12,15,20), got {output.shape}")
    corners, peaks, ambiguity = [], [], []
    for field in output[:4]:
        y, x = np.unravel_index(np.argmax(field), field.shape)
        corners.append((8.0 * (x + 0.5) - 0.5, 8.0 * (y + 0.5) - 0.5))
        peaks.append(float(field[y, x]))
        competing = field.copy()
        competing[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -np.inf
        ambiguity.append(float(field[y, x] - np.max(competing)))
    offset_scores = output[4:8].mean(axis=(1, 2))
    offsets = OFFSET_MIN + (
        np.clip(offset_scores, -SCORE_LIMIT, SCORE_LIMIT) + SCORE_LIMIT
    ) / (2.0 * SCORE_LIMIT) * (OFFSET_MAX - OFFSET_MIN)
    return DecodedSequentialOutput(
        corners_xy_crop=np.asarray(corners, np.float32),
        corner_peak_scores=np.asarray(peaks, np.float32),
        corner_ambiguity_margins=np.asarray(ambiguity, np.float32),
        offsets_m=offsets.astype(np.float32),
        confidence_scores=output[8:12].mean(axis=(1, 2)).astype(np.float32),
    )


def validate_gate_candidate(
    decoded: DecodedSequentialOutput,
    *,
    peak_min: float = 0.0,
    ambiguity_margin_min: float = 0.5,
    minimum_area_px2: float = 100.0,
    maximum_side_ratio: float = 6.0,
) -> bool:
    corners = np.asarray(decoded.corners_xy_crop, np.float32)
    if (
        corners.shape != (4, 2)
        or not np.isfinite(corners).all()
        or (decoded.corner_peak_scores < peak_min).any()
        or (decoded.corner_ambiguity_margins < ambiguity_margin_min).any()
    ):
        return False
    contour = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    area = abs(cv2.contourArea(corners))
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    return bool(
        area >= minimum_area_px2
        and sides.min() > 0
        and sides.max() / sides.min() <= maximum_side_ratio
    )


def estimate_gate_pose(
    decoded: DecodedSequentialOutput,
    camera_matrix_full160: np.ndarray,
    distortion: np.ndarray,
    *,
    label_centerline_extent_m: float,
    crop_top: float = 20.0,
    maximum_reprojection_error_px: float = 3.0,
) -> GatePose | None:
    if not validate_gate_candidate(decoded):
        return None
    half = float(label_centerline_extent_m) / 2.0
    object_points = np.asarray(
        [(-half, -half, 0), (half, -half, 0),
         (half, half, 0), (-half, half, 0)],
        np.float32,
    )
    image_points = decoded.corners_xy_crop.astype(np.float32).copy()
    image_points[:, 1] += crop_top
    success, rotation, translation = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(camera_matrix_full160, np.float64),
        np.asarray(distortion, np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or float(translation[2, 0]) <= 0:
        return None
    reprojected = cv2.projectPoints(
        object_points, rotation, translation,
        np.asarray(camera_matrix_full160, np.float64),
        np.asarray(distortion, np.float64),
    )[0].reshape(-1, 2)
    error = float(np.linalg.norm(reprojected - image_points, axis=1).mean())
    if error > maximum_reprojection_error_px:
        return None
    return GatePose(rotation[:, 0], translation[:, 0], error)


def effective_offsets(
    offsets_m: np.ndarray,
    confidence_scores: np.ndarray,
    *,
    confidence_min: float = 0.0,
    drone_radius_m: float = 0.10,
    tracking_margin_m: float = 0.05,
    body_speed_mps: float = 0.0,
    latency_s: float = 0.0,
    perception_margin_min_m: float = 0.03,
    perception_margin_gain_m: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(offsets_m, np.float64)
    confidence = np.asarray(confidence_scores, np.float64)
    accepted = confidence >= confidence_min
    confidence_unit = np.clip(
        (confidence + SCORE_LIMIT) / (2.0 * SCORE_LIMIT), 0.0, 1.0
    )
    perception_margin = perception_margin_min_m + perception_margin_gain_m * (
        1.0 - confidence_unit
    )
    rho = (
        drone_radius_m
        + tracking_margin_m
        + max(float(body_speed_mps), 0.0) * max(float(latency_s), 0.0)
        + perception_margin
    )
    return np.maximum(offsets - rho, 0.0), accepted


def body_offsets_to_world_halfspaces(
    effective_offsets_m: np.ndarray,
    accepted: np.ndarray,
    rotation_world_from_body: np.ndarray,
    current_position_world_m: np.ndarray,
) -> WorldHalfSpaces:
    rotation = np.asarray(rotation_world_from_body, np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_from_body must be 3x3")
    position = np.asarray(current_position_world_m, np.float64)
    normals_world = (rotation @ fixed_normals_body().T).T
    offsets = np.asarray(effective_offsets_m, np.float64)
    bounds = offsets + normals_world @ position
    return WorldHalfSpaces(
        normals_world=normals_world,
        bounds_world=bounds,
        accepted=np.asarray(accepted, bool),
        effective_offsets_m=offsets,
    )


def active_knot_mask(
    constraints: WorldHalfSpaces,
    nominal_positions_world_m: np.ndarray,
    activation_epsilon_m: float = 0.10,
) -> np.ndarray:
    positions = np.asarray(nominal_positions_world_m, np.float64)
    residual = positions @ constraints.normals_world.T - constraints.bounds_world
    return (residual > -activation_epsilon_m) & constraints.accepted[None, :]


def choose_reference_direction(
    offsets_m: np.ndarray,
    confidence_scores: np.ndarray,
    preferred_direction_body: np.ndarray,
    previous_direction_body: np.ndarray,
    dynamic_penalty: np.ndarray,
    *,
    weights: tuple[float, float, float, float] = (1.0, 0.6, 0.2, 0.4),
    confidence_min: float = 0.0,
) -> int | None:
    normals = fixed_normals_body()
    preferred = np.asarray(preferred_direction_body, np.float64)
    previous = np.asarray(previous_direction_body, np.float64)
    dynamic = np.asarray(dynamic_penalty, np.float64)
    confidence = np.asarray(confidence_scores, np.float64)
    valid = confidence >= confidence_min
    if not valid.any():
        return None
    confidence_unit = np.clip((confidence + SCORE_LIMIT) / (2 * SCORE_LIMIT), 0, 1)
    adjusted_distance = np.asarray(offsets_m, np.float64) * confidence_unit
    wd, wg, wh, wv = weights
    score = (
        wd * adjusted_distance
        + wg * (normals @ preferred)
        + wh * (normals @ previous)
        - wv * dynamic
    )
    score[~valid] = -np.inf
    return int(np.argmax(score))
