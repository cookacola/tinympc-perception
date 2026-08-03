"""Authoritative fixed-course gate geometry and camera-frame projection."""

from __future__ import annotations

import cv2
import numpy as np


GATE_CENTERS_WORLD = np.asarray(
    [(-1.30, -0.45, 0.55), (1.30, 0.45, 0.55)], dtype=np.float64
)
GATE_INNER_OPENING_M = 0.45
GATE_OUTER_EDGE_M = 0.66
GATE_LABEL_CENTERLINE_M = (GATE_INNER_OPENING_M + GATE_OUTER_EDGE_M) / 2.0

# Right-handed gate frame: +x is world +y, +y is world -z (image-like down
# near a front view), and +z is world -x.
ROTATION_WORLD_FROM_GATE = np.asarray(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def camera_rotation_world_to_camera(eye, target) -> np.ndarray:
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack((right, -up, forward))


def gate_projection(
    gate_index: int,
    eye,
    target,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    label_extent_m: float = GATE_LABEL_CENTERLINE_M,
    gate_centers_world: np.ndarray = GATE_CENTERS_WORLD,
) -> dict[str, np.ndarray]:
    half = label_extent_m / 2.0
    local = np.asarray(
        [(-half, -half, 0), (half, -half, 0),
         (half, half, 0), (-half, half, 0)],
        dtype=np.float64,
    )
    center = np.asarray(gate_centers_world, np.float64)[gate_index]
    world = center + (ROTATION_WORLD_FROM_GATE @ local.T).T
    rotation_camera_from_world = camera_rotation_world_to_camera(eye, target)
    translation = rotation_camera_from_world @ (center - np.asarray(eye))
    rotation_camera_from_gate = (
        rotation_camera_from_world @ ROTATION_WORLD_FROM_GATE
    )
    camera_points = (
        rotation_camera_from_gate @ local.T
    ).T + translation
    pixels = cv2.projectPoints(
        camera_points,
        np.zeros(3),
        np.zeros(3),
        np.asarray(camera_matrix, np.float64),
        np.asarray(distortion, np.float64),
    )[0][:, 0]
    # Preserve object-point correspondence while returning semantic image
    # order TL, TR, BR, BL.
    by_y = np.argsort(pixels[:, 1])
    top = by_y[:2][np.argsort(pixels[by_y[:2], 0])]
    bottom = by_y[2:][np.argsort(pixels[by_y[2:], 0])[::-1]]
    order = np.concatenate((top, bottom))
    return {
        "pixels_ordered": pixels[order],
        "object_points_ordered": local[order],
        "rotation_camera_from_gate": rotation_camera_from_gate,
        "translation_camera_from_gate": translation,
    }


def associate_gate(
    corners_xy160: np.ndarray,
    eye,
    target,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    inner_opening_m: float = GATE_INNER_OPENING_M,
    outer_edge_m: float = GATE_OUTER_EDGE_M,
    gate_centers_world: np.ndarray = GATE_CENTERS_WORLD,
) -> tuple[int, dict[str, np.ndarray], float]:
    # The semantic hole identifies the gate instance at the inner aperture.
    # The returned points follow the real annotation convention: the frame
    # centerline halfway between the inner opening and the outer edge.
    inner_candidates = [
        gate_projection(
            index, eye, target, camera_matrix, distortion,
            label_extent_m=inner_opening_m,
            gate_centers_world=gate_centers_world,
        )
        for index in range(len(gate_centers_world))
    ]
    errors = [
        np.linalg.norm(
            candidate["pixels_ordered"] - np.asarray(corners_xy160), axis=1
        ).mean()
        for candidate in inner_candidates
    ]
    index = int(np.argmin(errors))
    centerline_m = (float(inner_opening_m) + float(outer_edge_m)) / 2.0
    labels = gate_projection(
        index, eye, target, camera_matrix, distortion,
        label_extent_m=centerline_m,
        gate_centers_world=gate_centers_world,
    )
    return index, labels, float(errors[index])


def rotation_error_degrees(predicted: np.ndarray, truth: np.ndarray) -> float:
    relative = np.asarray(predicted) @ np.asarray(truth).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
