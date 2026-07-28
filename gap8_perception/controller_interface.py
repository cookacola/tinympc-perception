"""Controller-facing conversion from perception maps to TinyMPC constraints."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CorridorConstraints:
    corners_xy160: np.ndarray
    gate_center_xy160: np.ndarray
    safe_score_40: np.ndarray
    corridor_mask_40: np.ndarray
    image_lines_40: np.ndarray
    camera_plane_normals: np.ndarray
    reference_plane_normals: np.ndarray
    A: np.ndarray
    b: np.ndarray
    confidence: float
    constraint_mode: str
    lookahead_time_s: float
    camera_center_reference: np.ndarray
    slack_indices: tuple[int, ...]
    slack_penalty: float


@dataclass
class SlackTelemetry:
    """Aggregate soft-constraint health across TinyMPC solves."""

    solves: int = 0
    maximum_slack: float = 0.0
    total_slack_cost: float = 0.0
    active_image_constraints: int = 0
    nearly_infeasible_solves: int = 0

    def update(self, diagnostics: dict[str, float | int], threshold: float = 1e-3):
        self.solves += 1
        maximum = float(diagnostics["maximum_slack"])
        self.maximum_slack = max(self.maximum_slack, maximum)
        self.total_slack_cost += float(diagnostics["total_slack_cost"])
        self.active_image_constraints += int(
            diagnostics["active_image_constraints"]
        )
        self.nearly_infeasible_solves += int(maximum > threshold)


def decode_corners(heatmaps: np.ndarray, threshold: float = 0.20):
    corners, confidence = [], []
    for channel in heatmaps:
        y, x = np.unravel_index(np.argmax(channel), channel.shape)
        score = float(channel[y, x])
        confidence.append(score)
        y0, y1 = max(0, y - 2), min(channel.shape[0], y + 3)
        x0, x1 = max(0, x - 2), min(channel.shape[1], x + 3)
        patch = channel[y0:y1, x0:x1].clip(1e-8)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        corners.append([(patch * xx).sum() / patch.sum(), (patch * yy).sum() / patch.sum()])
    confidence = np.asarray(confidence, dtype=np.float32)
    return np.asarray(corners, dtype=np.float32) * 4.0, confidence, bool(
        np.all(confidence >= threshold)
    )


def connected_safe_corridor(
    gate_probability_40: np.ndarray,
    danger_probability_20: np.ndarray,
    gate_center_40: np.ndarray,
    threshold: float = 0.20,
    pixel_margin_160: float = 0.0,
):
    danger40 = cv2.resize(
        danger_probability_20, (40, 40), interpolation=cv2.INTER_LINEAR
    )
    score = gate_probability_40 * (1.0 - danger40)
    binary = (score >= threshold).astype(np.uint8)
    margin40 = int(np.ceil(max(pixel_margin_160, 0.0) / 4.0))
    if margin40:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin40 + 1, 2 * margin40 + 1)
        )
        binary = cv2.erode(binary, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return score, np.zeros_like(binary)
    x = int(np.clip(round(gate_center_40[0]), 0, 39))
    y = int(np.clip(round(gate_center_40[1]), 0, 39))
    label = int(labels[y, x])
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return score, (labels == label).astype(np.uint8)


def choose_pixel_margin_160(
    base_pixels: float = 2.0,
    *,
    range_bin: str | None = None,
    uncertainty: float = 0.0,
    calibration_error_pixels: float = 0.0,
    latency_s: float = 0.0,
    speed_mps: float = 0.0,
    focal_length_pixels: float | None = None,
    conservative_range_m: float | None = None,
    drone_radius_m: float = 0.10,
) -> float:
    """Combine projected radius, range-bin, uncertainty, calibration, and latency."""
    range_bin_margin = {None: 0.0, "far": 1.0, "medium": 3.0, "near": 6.0}
    if range_bin not in range_bin_margin:
        raise ValueError("range_bin must be near, medium, far, or None")
    projected_radius = 0.0
    if focal_length_pixels is not None and conservative_range_m is not None:
        projected_radius = (
            focal_length_pixels * drone_radius_m / max(conservative_range_m, 1e-3)
        )
    latency_margin = 0.0
    if focal_length_pixels is not None and conservative_range_m is not None:
        latency_margin = (
            focal_length_pixels * speed_mps * latency_s
            / max(conservative_range_m, 1e-3)
        )
    return float(
        base_pixels
        + range_bin_margin[range_bin]
        + max(uncertainty, 0.0) * 4.0
        + max(calibration_error_pixels, 0.0)
        + projected_radius
        + latency_margin
    )


def fit_corridor_lines(mask: np.ndarray) -> np.ndarray:
    ys, left, right = [], [], []
    for y in range(mask.shape[0]):
        xs = np.flatnonzero(mask[y])
        if len(xs):
            ys.append(y)
            left.append(xs.min())
            right.append(xs.max())
    if len(ys) < 2:
        raise ValueError("safe corridor has insufficient vertical support")
    left_fit = np.polyfit(ys, left, 1)   # x = a*y+b
    right_fit = np.polyfit(ys, right, 1)
    return np.asarray(
        [
            [1.0, -left_fit[0], -left_fit[1]],
            [1.0, -right_fit[0], -right_fit[1]],
        ],
        dtype=np.float64,
    )


def image_lines_to_planes(
    lines_40: np.ndarray,
    intrinsics_160: np.ndarray,
    rotation_reference_from_camera: np.ndarray,
    feasible_point_40: np.ndarray,
    distortion_coefficients: np.ndarray | None = None,
):
    normals_camera = []
    for a, b, c in lines_40:
        if distortion_coefficients is None:
            # a*u40+b*v40+c=0 with u160=4*u40 maps to
            # l160=[a/4,b/4,c], then n_c=K^T*l160.
            normals_camera.append(
                pixel_line_to_angular_plane(
                    np.array([a / 4.0, b / 4.0, c]),
                    intrinsics_160,
                )
            )
            continue
        endpoints40 = np.array(
            [[-(b * 0.0 + c) / a, 0.0], [-(b * 39.0 + c) / a, 39.0]],
            dtype=np.float64,
        )
        endpoints160 = endpoints40 * 4.0
        normalized = cv2.undistortPoints(
            endpoints160.reshape(-1, 1, 2),
            intrinsics_160,
            None if distortion_coefficients is None else distortion_coefficients,
        ).reshape(-1, 2)
        ray0 = np.array([normalized[0, 0], normalized[0, 1], 1.0])
        ray1 = np.array([normalized[1, 0], normalized[1, 1], 1.0])
        normals_camera.append(np.cross(ray0, ray1))
    normals_camera = np.asarray(normals_camera)
    center_normalized = cv2.undistortPoints(
        (feasible_point_40[None, None] * 4.0).astype(np.float64),
        intrinsics_160,
        None if distortion_coefficients is None else distortion_coefficients,
    )[0, 0]
    ray = np.array([center_normalized[0], center_normalized[1], 1.0])
    for index in range(len(normals_camera)):
        normals_camera[index] /= np.linalg.norm(normals_camera[index]) + 1e-12
        if normals_camera[index] @ ray < 0:
            normals_camera[index] *= -1
    normals_reference = (
        rotation_reference_from_camera @ normals_camera.T
    ).T
    return normals_camera, normals_reference


def pixel_line_to_angular_plane(
    line_pixels: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Return normalized camera-plane normal n=K^T l for an undistorted line."""
    normal = np.asarray(intrinsics, np.float64).T @ np.asarray(
        line_pixels, np.float64
    )
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("degenerate image line")
    return normal / norm


def lookahead_inequality_row(
    normal_reference: np.ndarray,
    state_dim: int,
    position_indices: tuple[int, int, int],
    velocity_indices: tuple[int, int, int],
    camera_center_reference: np.ndarray,
    lookahead_time_s: float,
    slack_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """Encode -n'p - tau*n'v - s <= -n'p_c0."""
    row = np.zeros(state_dim, dtype=np.float64)
    normal = np.asarray(normal_reference, np.float64)
    row[list(position_indices)] = -normal
    row[list(velocity_indices)] = -lookahead_time_s * normal
    if slack_index is not None:
        row[slack_index] = -1.0
    bound = -float(normal @ np.asarray(camera_center_reference, np.float64))
    return row, bound


def constraint_slack_diagnostics(
    A: np.ndarray,
    b: np.ndarray,
    state: np.ndarray,
    slack_penalty: float,
) -> dict[str, float | int]:
    """Report the nonnegative slack needed to make image rows feasible."""
    required = np.maximum(np.asarray(A) @ np.asarray(state) - np.asarray(b), 0.0)
    return {
        "maximum_slack": float(required.max(initial=0.0)),
        "total_slack_cost": float(slack_penalty * np.square(required).sum()),
        "active_image_constraints": int((required > 0).sum()),
        "nearly_infeasible_constraints": int((required > 1e-4).sum()),
    }


def selected_horizon_constraint_rows(
    constraints: CorridorConstraints,
    selected_steps: list[int] | tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return frozen image-plane rows for selected future steps only."""
    if any(step <= 0 for step in selected_steps):
        raise ValueError("image constraints must exclude current step 0")
    return {
        int(step): (constraints.A.copy(), constraints.b.copy())
        for step in selected_steps
    }


def nonnegative_slack_state_bounds(
    state_dim: int, slack_indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return TinyMPC state bounds that enforce every image slack s>=0."""
    lower = np.full(state_dim, -np.inf, dtype=np.float64)
    upper = np.full(state_dim, np.inf, dtype=np.float64)
    lower[list(slack_indices)] = 0.0
    return lower, upper


def metric_obstacle_halfspace(
    obstacle_normal_reference: np.ndarray,
    *,
    boundary_point_reference: np.ndarray | None = None,
    conservative_range_m: float | None = None,
    camera_center_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Build n'p >= offset only when a metric point or range is supplied."""
    normal = np.asarray(obstacle_normal_reference, np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    if boundary_point_reference is None:
        if conservative_range_m is None or camera_center_reference is None:
            raise ValueError(
                "metric half-space requires a boundary point or both "
                "conservative range and camera center"
            )
        boundary_point_reference = (
            np.asarray(camera_center_reference, np.float64)
            + conservative_range_m * normal
        )
    # Standard <= form for the feasible side n'p >= n'p_boundary.
    return -normal, -float(normal @ np.asarray(boundary_point_reference))


def build_tinympc_constraints(
    corner_heatmaps_40: np.ndarray,
    danger_probability_20: np.ndarray,
    gate_probability_40: np.ndarray,
    intrinsics_160: np.ndarray,
    rotation_reference_from_camera: np.ndarray,
    state_dim: int,
    velocity_indices: tuple[int, int, int],
    distortion_coefficients: np.ndarray | None = None,
    *,
    constraint_mode: str = "lookahead",
    position_indices: tuple[int, int, int] = (0, 1, 2),
    camera_center_reference: np.ndarray | None = None,
    lookahead_time_s: float = 0.3,
    pixel_margin_160: float = 4.0,
    slack_indices: tuple[int, int] | None = None,
    slack_penalty: float = 1e4,
) -> CorridorConstraints:
    corners160, corner_scores, valid = decode_corners(corner_heatmaps_40)
    if not valid:
        raise ValueError("corner confidence below threshold")
    center160 = corners160.mean(axis=0)
    center40 = center160 / 4.0
    score, corridor = connected_safe_corridor(
        gate_probability_40,
        danger_probability_20,
        center40,
        pixel_margin_160=pixel_margin_160,
    )
    lines = fit_corridor_lines(corridor)
    camera_normals, reference_normals = image_lines_to_planes(
        lines,
        intrinsics_160,
        rotation_reference_from_camera,
        center40,
        distortion_coefficients,
    )
    camera_center_reference = (
        np.zeros(3, dtype=np.float64)
        if camera_center_reference is None
        else np.asarray(camera_center_reference, dtype=np.float64)
    )
    if constraint_mode not in {"lookahead", "velocity"}:
        raise ValueError("constraint_mode must be 'lookahead' or 'velocity'")
    if slack_indices is not None and len(slack_indices) != len(reference_normals):
        raise ValueError("one slack index is required per image constraint")
    rows, bounds = [], []
    for index, normal in enumerate(reference_normals):
        slack_index = None if slack_indices is None else slack_indices[index]
        if constraint_mode == "lookahead":
            row, bound = lookahead_inequality_row(
                normal,
                state_dim,
                position_indices,
                velocity_indices,
                camera_center_reference,
                lookahead_time_s,
                slack_index,
            )
        else:
            row = np.zeros(state_dim, dtype=np.float64)
            row[list(velocity_indices)] = -normal
            if slack_index is not None:
                row[slack_index] = -1.0
            bound = 0.0
        rows.append(row)
        bounds.append(bound)
    A = np.asarray(rows)
    b = np.asarray(bounds)
    confidence = float(
        min(corner_scores.min(), gate_probability_40[corridor > 0].mean())
    )
    return CorridorConstraints(
        corners_xy160=corners160,
        gate_center_xy160=center160,
        safe_score_40=score,
        corridor_mask_40=corridor,
        image_lines_40=lines,
        camera_plane_normals=camera_normals,
        reference_plane_normals=reference_normals,
        A=A,
        b=b,
        confidence=confidence,
        constraint_mode=constraint_mode,
        lookahead_time_s=lookahead_time_s,
        camera_center_reference=camera_center_reference,
        slack_indices=tuple() if slack_indices is None else slack_indices,
        slack_penalty=slack_penalty,
    )
