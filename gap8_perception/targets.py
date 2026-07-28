"""Reproducible bootstrap targets from existing depth and semantic annotations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
COLLISION_CLASSES = {"obstacle", "boundary", "gate", "lab_clutter"}


def reliable_gate_corner_view(
    corners_xy: np.ndarray,
    *,
    minimum_opening_area_px2: float = 100.0,
    minimum_side_px: float = 5.0,
    maximum_side_ratio: float = 6.0,
) -> bool:
    """Whether a projected gate is resolved well enough for corner regression.

    A closed semantic opening alone is insufficient: a distant or nearly
    edge-on gate can still form a four-sided raster hole while its individual
    corners are not visually localizable at HM01B0 resolution.
    """
    corners = np.asarray(corners_xy, dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    rolled = np.roll(corners, -1, axis=0)
    area = 0.5 * abs(
        float(np.sum(corners[:, 0] * rolled[:, 1] - corners[:, 1] * rolled[:, 0]))
    )
    sides = np.linalg.norm(corners - rolled, axis=1)
    shortest = float(sides.min())
    longest = float(sides.max())
    return (
        area >= minimum_opening_area_px2
        and shortest >= minimum_side_px
        and longest / max(shortest, 1e-6) <= maximum_side_ratio
    )


def class_mask(semantic_path: Path, labels_path: Path, classes: set[str]) -> np.ndarray:
    semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
    if semantic is None:
        raise ValueError(f"cannot read {semantic_path}")
    labels = json.loads(labels_path.read_text())
    raw_ids = [
        int(raw)
        for raw, fields in labels.items()
        if str(fields.get("class", "")).lower() in classes
    ]
    return np.isin(semantic, raw_ids).astype(np.uint8)


def gate_opening_and_corners(gate_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract the largest enclosed gate opening and ordered raster corners.

    This intentionally returns invalid for partial/occluded gates whose opening
    is not a closed hole in the semantic gate-frame mask.
    """
    contours, hierarchy = cv2.findContours(
        gate_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    opening = np.zeros_like(gate_frame, dtype=np.uint8)
    corners = np.full((4, 2), np.nan, dtype=np.float32)
    if hierarchy is None:
        return opening, corners, False
    holes = [
        contour
        for contour, node in zip(contours, hierarchy[0])
        if node[3] >= 0 and cv2.contourArea(contour) >= 16
    ]
    if not holes:
        return opening, corners, False
    hole = max(holes, key=cv2.contourArea)
    cv2.drawContours(opening, [hole], -1, 1, thickness=cv2.FILLED)

    perimeter = cv2.arcLength(hole, True)
    polygon = cv2.approxPolyDP(hole, 0.04 * perimeter, True).reshape(-1, 2)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(hole)).astype(np.float32)
    else:
        polygon = polygon.astype(np.float32)
    center = polygon.mean(axis=0)
    top = polygon[polygon[:, 1] <= center[1]]
    bottom = polygon[polygon[:, 1] > center[1]]
    if len(top) != 2 or len(bottom) != 2:
        return opening, corners, False
    corners[:] = [
        top[np.argmin(top[:, 0])],
        top[np.argmax(top[:, 0])],
        bottom[np.argmax(bottom[:, 0])],
        bottom[np.argmin(bottom[:, 0])],
    ]
    return opening, corners, reliable_gate_corner_view(corners)


def danger_target(
    depth_mm: np.ndarray,
    collision_mask: np.ndarray,
    output_size: int = 20,
    immediate_m: float = 1.0,
    caution_m: float = 2.5,
    focal_length_px: float = 89.1558392549,
    clearance_m: float = 0.20,
) -> np.ndarray:
    """Create continuous conservative risk from collision semantics and depth.

    Motion state is absent in the source set, so this is explicitly a
    depth-band bootstrap target, not a rollout-derived collision label.
    """
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid_collision = (collision_mask > 0) & (depth_m > 0)
    risk = np.zeros(depth_m.shape, dtype=np.float32)
    risk[valid_collision & (depth_m <= immediate_m)] = 1.0
    caution = valid_collision & (depth_m > immediate_m) & (depth_m < caution_m)
    risk[caution] = 0.25 + 0.75 * (
        (caution_m - depth_m[caution]) / (caution_m - immediate_m)
    )
    # Project drone radius+safety margin at the collision depth. Near objects
    # receive a larger image-space expansion than distant objects.
    expanded = np.zeros_like(risk)
    for near, far in ((0.05, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, caution_m)):
        band = valid_collision & (depth_m >= near) & (depth_m < far)
        if not band.any():
            continue
        representative_depth = max(near, float(np.median(depth_m[band])))
        radius_px = int(np.ceil(focal_length_px * clearance_m / representative_depth))
        radius_px = int(np.clip(radius_px, 1, 40))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
        )
        expanded = np.maximum(expanded, cv2.dilate(risk * band, kernel))
    risk = expanded
    cell = depth_m.shape[0] // output_size
    return risk.reshape(output_size, cell, output_size, cell).max(axis=(1, 3))


def gaussian_heatmaps(
    corners_xy: np.ndarray, valid: bool, size: int = 40, sigma: float = 1.25
) -> np.ndarray:
    heatmaps = np.zeros((4, size, size), dtype=np.float32)
    if not valid:
        return heatmaps
    yy, xx = np.mgrid[:size, :size]
    scaled = corners_xy * (size / 160.0)
    for channel, (x, y) in enumerate(scaled):
        heatmaps[channel] = np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    return heatmaps
