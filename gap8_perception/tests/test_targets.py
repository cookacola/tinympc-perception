import numpy as np

from gap8_perception.targets import reliable_gate_corner_view


def test_reliable_gate_corner_view_accepts_resolved_square():
    corners = np.array([[40, 40], [120, 40], [120, 120], [40, 120]])
    assert reliable_gate_corner_view(corners)


def test_reliable_gate_corner_view_rejects_distant_gate():
    corners = np.array([[78, 78], [82, 78], [82, 82], [78, 82]])
    assert not reliable_gate_corner_view(corners)


def test_reliable_gate_corner_view_rejects_extreme_angle():
    corners = np.array([[50, 50], [110, 50], [111, 53], [49, 53]])
    assert not reliable_gate_corner_view(corners)
