import numpy as np

from gap8_perception.controller_sequential import (
    active_knot_mask,
    body_offsets_to_world_halfspaces,
    choose_reference_direction,
    decode_output,
    estimate_gate_pose,
    effective_offsets,
    validate_gate_candidate,
)


def test_decode_matches_single_tensor_contract():
    output = np.zeros((12, 15, 20), np.float32)
    output[0, 2, 3] = 5
    output[4:8] = 0
    decoded = decode_output(output)
    np.testing.assert_allclose(decoded.corners_xy_crop[0], [27.5, 19.5])
    np.testing.assert_allclose(decoded.offsets_m, 3.0)


def test_low_confidence_is_rejected_and_never_free_space():
    effective, accepted = effective_offsets(
        np.full(4, 6.0), np.asarray([1.0, -0.1, 2.0, -3.0])
    )
    np.testing.assert_array_equal(accepted, [True, False, True, False])
    assert np.all(effective < 6.0)


def test_world_planes_and_knot_activation_follow_design_equations():
    constraints = body_offsets_to_world_halfspaces(
        [1.0, 1.0, 1.0, 1.0], [True] * 4, np.eye(3), [2.0, 3.0, 0.0]
    )
    current = np.asarray([[2.0, 3.0, 0.0]])
    assert not active_knot_mask(constraints, current).any()
    near_boundary = current + constraints.normals_world[[0]] * 0.95
    assert active_knot_mask(constraints, near_boundary)[0, 0]


def test_reference_selection_ignores_low_confidence_direction():
    selected = choose_reference_direction(
        [6.0, 2.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], np.zeros(4),
    )
    assert selected != 0


def test_gate_geometry_validation_rejects_ambiguous_or_nonconvex_candidates():
    output = np.full((12, 15, 20), -6.0, np.float32)
    for channel, (y, x) in enumerate(((3, 4), (3, 14), (11, 14), (11, 4))):
        output[channel, y, x] = 6.0
    decoded = decode_output(output)
    assert validate_gate_candidate(decoded)
    output[0, 13, 18] = 5.9
    assert not validate_gate_candidate(decode_output(output))


def test_centerline_corner_pnp_round_trip():
    K = np.asarray([[100.0, 0, 80.0], [0, 100.0, 60.0], [0, 0, 1]])
    output = np.full((12, 15, 20), -6.0, np.float32)
    for channel, (y, x) in enumerate(((4, 6), (4, 13), (10, 13), (10, 6))):
        output[channel, y, x] = 6.0
    pose = estimate_gate_pose(
        decode_output(output), K, np.zeros(5), label_centerline_extent_m=0.62,
        maximum_reprojection_error_px=10.0,
    )
    assert pose is not None and pose.translation_vector_m[2] > 0
