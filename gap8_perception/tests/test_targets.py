import numpy as np

from gap8_perception.targets import gate_opening_and_corners, reliable_gate_corner_view


def test_reliable_gate_corner_view_accepts_resolved_square():
    corners = np.array([[40, 40], [120, 40], [120, 120], [40, 120]])
    assert reliable_gate_corner_view(corners)


def test_reliable_gate_corner_view_rejects_distant_gate():
    corners = np.array([[78, 78], [82, 78], [82, 82], [78, 82]])
    assert not reliable_gate_corner_view(corners)


def test_reliable_gate_corner_view_rejects_extreme_angle():
    corners = np.array([[50, 50], [110, 50], [111, 53], [49, 53]])
    assert not reliable_gate_corner_view(corners)


def test_gate_corners_are_midway_between_inner_and_outer_frame():
    gate = np.zeros((160, 160), np.uint8)
    gate[20:141, 20:141] = 1
    gate[50:111, 50:111] = 0

    opening, corners, valid = gate_opening_and_corners(gate)

    assert valid
    assert opening[80, 80] == 1
    expected = np.asarray(
        [[35, 35], [125, 35], [125, 125], [35, 125]], np.float32
    )
    np.testing.assert_allclose(corners, expected, atol=1.0)
