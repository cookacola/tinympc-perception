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
    recovered_corner: int | None = None


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
    allow_three_corner_recovery: bool = True,
) -> GateDecision:
    corners = np.asarray(corners_xy160, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if corners.shape != (4, 2) or confidence.shape != (4,):
        return GateDecision(False, corners, confidence, "shape")
    if not np.isfinite(corners).all() or not np.isfinite(confidence).all():
        return GateDecision(False, corners, confidence, "nonfinite")
    confident = confidence >= confidence_threshold
    recovered_corner = None
    if confident.sum() == 3 and allow_three_corner_recovery:
        recovered_corner = int(np.flatnonzero(~confident)[0])
        # Ordered TL/TR/BR/BL corners obey this affine quadrilateral relation.
        # It is exact for a parallelogram and a conservative approximation for
        # the modest projective distortion expected from a race gate.
        opposite = (recovered_corner + 2) & 3
        previous = (recovered_corner - 1) & 3
        following = (recovered_corner + 1) & 3
        corners = corners.copy()
        corners[recovered_corner] = (
            corners[previous] + corners[following] - corners[opposite]
        )
        # Never extrapolate a missing corner outside the observed CNN crop.
        if not (
            0.0 <= corners[recovered_corner, 0] < 160.0
            and 20.0 <= corners[recovered_corner, 1] < 140.0
        ):
            return GateDecision(
                False, corners, confidence, "recovered_out_of_bounds",
                recovered_corner,
            )
    elif not confident.all():
        return GateDecision(False, corners, confidence, "confidence")
    # Required semantic order: TL, TR, BR, BL.
    if not (
        corners[0, 0] < corners[1, 0]
        and corners[3, 0] < corners[2, 0]
        and corners[0, 1] < corners[3, 1]
        and corners[1, 1] < corners[2, 1]
    ):
        return GateDecision(
            False, corners, confidence, "ordering", recovered_corner
        )
    contour = corners.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return GateDecision(
            False, corners, confidence, "nonconvex", recovered_corner
        )
    area = abs(float(cv2.contourArea(contour)))
    if area < minimum_area_px2:
        return GateDecision(False, corners, confidence, "area", recovered_corner)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if float(sides.max() / max(sides.min(), 1e-6)) > maximum_side_ratio:
        return GateDecision(
            False, corners, confidence, "side_ratio", recovered_corner
        )
    reason = "accepted_three_corners" if recovered_corner is not None else "accepted"
    return GateDecision(True, corners, confidence, reason, recovered_corner)


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
    recovery_inset = 1 if decision.recovered_corner is not None else 0
    polygon -= vectors / lengths * float(inset_cells + recovery_inset)
    mask = np.zeros((15, 20), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    output = danger.copy()
    output[mask > 0] = np.minimum(output[mask > 0], residual_floor)
    return output
