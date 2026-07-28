import numpy as np

from gap8_perception.postprocess_stdc import (
    GateDecision,
    gate_override_danger,
    validate_gate_geometry,
)


def test_valid_gate_overrides_only_inset_opening():
    corners = np.asarray([[40, 40], [120, 40], [120, 120], [40, 120]], np.float32)
    decision = validate_gate_geometry(corners, np.full(4, 0.9, np.float32))
    output = gate_override_danger(np.ones((15, 20), np.float32), decision)
    assert decision.accepted
    assert output[7, 10] == 0.05
    assert output[0, 0] == 1.0


def test_low_confidence_never_overrides_danger():
    corners = np.asarray([[40, 40], [120, 40], [120, 120], [40, 120]], np.float32)
    decision = validate_gate_geometry(
        corners, np.asarray([0.9, 0.9, 0.1, 0.9], np.float32)
    )
    danger = np.ones((15, 20), np.float32)
    assert not decision.accepted
    assert np.array_equal(gate_override_danger(danger, decision), danger)


def test_impossible_order_and_extreme_geometry_are_rejected():
    confidence = np.full(4, 0.9, np.float32)
    crossed = np.asarray([[40, 40], [120, 120], [120, 40], [40, 120]], np.float32)
    thin = np.asarray([[40, 40], [140, 40], [140, 45], [40, 45]], np.float32)
    assert not validate_gate_geometry(crossed, confidence).accepted
    assert not validate_gate_geometry(thin, confidence).accepted


def test_rejected_decision_is_safe_noop():
    decision = GateDecision(
        False, np.zeros((4, 2), np.float32), np.zeros(4, np.float32), "test"
    )
    danger = np.random.default_rng(7).random((15, 20), dtype=np.float32)
    assert np.array_equal(gate_override_danger(danger, decision), danger)
