import cv2
import numpy as np

from gap8_perception.controller_interface import (
    build_tinympc_constraints,
    choose_pixel_margin_160,
    constraint_slack_diagnostics,
    image_lines_to_planes,
    lookahead_inequality_row,
    metric_obstacle_halfspace,
    nonnegative_slack_state_bounds,
    pixel_line_to_angular_plane,
    selected_horizon_constraint_rows,
)
from gap8_perception.rollout_targets import simulate_candidate_rollouts


def synthetic_outputs():
    corners = np.zeros((4, 40, 40), np.float32)
    for channel, (x, y) in enumerate(((12, 10), (28, 10), (28, 30), (12, 30))):
        corners[channel, y, x] = 1
    danger = np.zeros((20, 20), np.float32)
    gate = np.zeros((40, 40), np.float32)
    gate[10:31, 12:29] = 1
    return corners, danger, gate


def test_gate_center_ray_is_feasible_and_dimensions_match():
    corners, danger, gate = synthetic_outputs()
    K = np.array([[100, 0, 80], [0, 100, 80], [0, 0, 1]], np.float64)
    result = build_tinympc_constraints(
        corners, danger, gate, K, np.eye(3), state_dim=12,
        velocity_indices=(3, 4, 5), constraint_mode="velocity",
    )
    ray = np.linalg.inv(K) @ np.array([*result.gate_center_xy160, 1.0])
    state = np.zeros(12)
    state[3:6] = ray
    assert result.A.shape == (2, 12)
    assert result.b.shape == (2,)
    assert np.all(result.A @ state <= result.b + 1e-9)


def test_camera_to_reference_rotation_is_applied():
    lines = np.array([[1, 0, -10], [-1, 0, 30]], np.float64)
    K = np.array([[100, 0, 80], [0, 100, 80], [0, 0, 1]], np.float64)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
    camera, reference = image_lines_to_planes(lines, K, rotation, np.array([20, 20]))
    assert np.allclose(reference, (rotation @ camera.T).T)


def test_projecting_violating_velocity_moves_it_to_safe_side():
    corners, danger, gate = synthetic_outputs()
    K = np.array([[100, 0, 80], [0, 100, 80], [0, 0, 1]], np.float64)
    result = build_tinympc_constraints(
        corners, danger, gate, K, np.eye(3), state_dim=6,
        velocity_indices=(0, 1, 2), constraint_mode="velocity",
    )
    state = np.zeros(6)
    row = result.A[0]
    state[:] = row
    violation = row @ state - result.b[0]
    assert violation > 0
    projected = state - (violation / (row @ row)) * row
    assert row @ projected <= result.b[0] + 1e-9


def test_user_calibration_distortion_path_preserves_feasible_center():
    corners, danger, gate = synthetic_outputs()
    K = np.array(
        [[89.1558392549, 0, 81.1038105230],
         [0, 89.4608171623, 73.3473030288],
         [0, 0, 1]],
        np.float64,
    )
    dist = np.array(
        [-0.0176448766, 0.0994132451, 0.0054432154, -0.0060400120, -0.1900189875]
    )
    result = build_tinympc_constraints(
        corners,
        danger,
        gate,
        K,
        np.eye(3),
        state_dim=6,
        velocity_indices=(0, 1, 2),
        distortion_coefficients=dist,
        constraint_mode="velocity",
    )
    center = cv2.undistortPoints(
        result.gate_center_xy160.reshape(1, 1, 2), K, dist
    )[0, 0]
    state = np.array([center[0], center[1], 1.0, 0, 0, 0])
    assert np.all(result.A @ state <= result.b + 1e-9)


def test_k_transpose_line_plane_contains_all_line_rays():
    K = np.array([[90, 0, 81], [0, 91, 73], [0, 0, 1]], np.float64)
    line = np.array([1.0, -0.5, -25.0])
    normal = pixel_line_to_angular_plane(line, K)
    for y in (0.0, 80.0, 159.0):
        x = 0.5 * y + 25.0
        ray = np.linalg.inv(K) @ np.array([x, y, 1.0])
        assert abs(normal @ ray) < 1e-10


def test_reversing_line_reverses_raw_plane_convention():
    K = np.array([[90, 0, 81], [0, 91, 73], [0, 0, 1]], np.float64)
    line = np.array([1.0, 0.0, -50.0])
    assert np.allclose(
        pixel_line_to_angular_plane(-line, K),
        -pixel_line_to_angular_plane(line, K),
    )


def test_lookahead_row_matches_state_order_and_danger_side_violates():
    n = np.array([1.0, 0.0, 0.0])
    row, bound = lookahead_inequality_row(
        n,
        state_dim=8,
        position_indices=(2, 3, 4),
        velocity_indices=(5, 6, 7),
        camera_center_reference=np.array([1.0, 0.0, 0.0]),
        lookahead_time_s=0.25,
    )
    assert np.allclose(row, [0, 0, -1, 0, 0, -0.25, 0, 0])
    assert bound == -1.0
    safe = np.zeros(8)
    safe[2], safe[5] = 1.0, 1.0
    danger_side = np.zeros(8)
    danger_side[2], danger_side[5] = 0.5, -1.0
    assert row @ safe <= bound
    assert row @ danger_side > bound


def test_slack_or_projection_restores_safe_angular_side():
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    state = np.array([0.4, 2.0])
    diagnostics = constraint_slack_diagnostics(A, b, state, 1000.0)
    assert diagnostics["maximum_slack"] == 0.4
    projected = state - ((A @ state - b) / (A @ A.T))[0] * A[0]
    assert (A @ projected <= b + 1e-12).all()


def test_identical_image_ray_different_range_time_has_different_urgency():
    K = np.array([[89.0, 0, 80], [0, 89.0, 80], [0, 0, 1]], np.float32)
    dist = np.zeros(5, np.float32)
    near_box = np.array([[[0.8, -0.2, -0.2], [1.0, 0.2, 0.2]]], np.float32)
    far_box = np.array([[[2.8, -0.2, -0.2], [3.0, 0.2, 0.2]]], np.float32)
    kwargs = dict(
        eye=[0, 0, 0], target=[1, 0, 0], camera_matrix=K,
        distortion=dist, speed_mps=3.0, horizon_s=1.0,
        drone_radius_m=0.05, safety_margin_m=0.0,
    )
    near = simulate_candidate_rollouts(**kwargs, boxes=near_box)
    far = simulate_candidate_rollouts(**kwargs, boxes=far_box)
    center = (10, 10)
    assert near["ttc_s"][center] < far["ttc_s"][center]
    assert near["urgency"][center] > far["urgency"][center]


def test_metric_plane_refuses_to_invent_an_offset_without_range():
    try:
        metric_obstacle_halfspace(np.array([1.0, 0.0, 0.0]))
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("metric half-space accepted no range or point")


def test_margin_combines_range_uncertainty_calibration_and_latency():
    far = choose_pixel_margin_160(
        range_bin="far", uncertainty=0.1, calibration_error_pixels=0.2,
        latency_s=0.08, speed_mps=1.0, focal_length_pixels=90,
        conservative_range_m=4.0,
    )
    near = choose_pixel_margin_160(
        range_bin="near", uncertainty=0.5, calibration_error_pixels=0.2,
        latency_s=0.08, speed_mps=5.0, focal_length_pixels=90,
        conservative_range_m=1.0,
    )
    assert near > far


def test_selected_horizon_rows_exclude_current_and_stay_frozen():
    corners, danger, gate = synthetic_outputs()
    K = np.array([[100, 0, 80], [0, 100, 80], [0, 0, 1]], np.float64)
    constraints = build_tinympc_constraints(
        corners, danger, gate, K, np.eye(3), state_dim=8,
        position_indices=(0, 1, 2), velocity_indices=(3, 4, 5),
        slack_indices=(6, 7),
    )
    rows = selected_horizon_constraint_rows(constraints, [1, 3, 5])
    assert set(rows) == {1, 3, 5}
    assert all(np.array_equal(A, constraints.A) for A, _ in rows.values())
    lower, upper = nonnegative_slack_state_bounds(8, (6, 7))
    assert np.all(lower[6:] == 0) and np.all(np.isposinf(upper[6:]))
    try:
        selected_horizon_constraint_rows(constraints, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("current-state image constraint was accepted")
