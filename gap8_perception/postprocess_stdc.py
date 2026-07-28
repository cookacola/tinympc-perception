"""Confidence and geometry gating for STDC corner/danger predictions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    corners_xy160: np.ndarray
    confidence: np.ndarray
    reason: str


def decode_corner_heatmaps(corner_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode ordered TL/TR/BR/BL maps into full 160x160 sensor coordinates."""
    heatmaps = np.asarray(corner_probability, dtype=np.float32)
    if heatmaps.shape != (4, 30, 40):
        raise ValueError(f"expected (4,30,40), got {heatmaps.shape}")
    corners = np.zeros((4, 2), dtype=np.float32)
    confidence = heatmaps.reshape(4, -1).max(axis=1)
    for channel, plane in enumerate(heatmaps):
        peak_y, peak_x = np.unravel_index(int(np.argmax(plane)), plane.shape)
        corners[channel] = ((peak_x + 0.5) * 4.0, (peak_y + 0.5) * 4.0 + 20.0)
    return corners, confidence


def validate_gate_geometry(
    corners_xy160: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float = 0.25,
    minimum_area_px2: float = 100.0,
    maximum_side_ratio: float = 6.0,
) -> GateDecision:
    corners = np.asarray(corners_xy160, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if corners.shape != (4, 2) or confidence.shape != (4,):
        return GateDecision(False, corners, confidence, "shape")
    if not np.isfinite(corners).all() or not np.isfinite(confidence).all():
        return GateDecision(False, corners, confidence, "nonfinite")
    if (confidence < confidence_threshold).any():
        return GateDecision(False, corners, confidence, "confidence")
    # Required semantic order: TL, TR, BR, BL.
    if not (
        corners[0, 0] < corners[1, 0]
        and corners[3, 0] < corners[2, 0]
        and corners[0, 1] < corners[3, 1]
        and corners[1, 1] < corners[2, 1]
    ):
        return GateDecision(False, corners, confidence, "ordering")
    contour = corners.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return GateDecision(False, corners, confidence, "nonconvex")
    area = abs(float(cv2.contourArea(contour)))
    if area < minimum_area_px2:
        return GateDecision(False, corners, confidence, "area")
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if float(sides.max() / max(sides.min(), 1e-6)) > maximum_side_ratio:
        return GateDecision(False, corners, confidence, "side_ratio")
    return GateDecision(True, corners, confidence, "accepted")


def gate_override_danger(
    danger_probability: np.ndarray,
    decision: GateDecision,
    *,
    inset_cells: int = 1,
    residual_floor: float = 0.05,
) -> np.ndarray:
    """Reduce danger only inside an accepted, inset gate quadrilateral.

    This implements the design document's two-head arbitration.  It never
    changes danger for a low-confidence or geometrically invalid gate.
    """
    danger = np.asarray(danger_probability, dtype=np.float32)
    if danger.shape != (15, 20):
        raise ValueError(f"expected (15,20), got {danger.shape}")
    if not decision.accepted:
        return danger.copy()
    polygon = decision.corners_xy160.copy()
    polygon[:, 0] *= 20.0 / 160.0
    polygon[:, 1] = (polygon[:, 1] - 20.0) * (15.0 / 120.0)
    center = polygon.mean(axis=0)
    vectors = polygon - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-6)
    polygon -= vectors / lengths * float(inset_cells)
    mask = np.zeros((15, 20), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    output = danger.copy()
    output[mask > 0] = np.minimum(output[mask > 0], residual_floor)
    return output
