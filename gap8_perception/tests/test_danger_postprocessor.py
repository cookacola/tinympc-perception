import numpy as np

from gap8_perception.danger_postprocessor import collision_probability_from_range


def test_same_image_range_becomes_more_dangerous_at_higher_speed():
    inverse_range = np.full((20, 20), 0.5, np.float32)  # 3 m
    base = np.ones_like(inverse_range)
    uncertainty = np.zeros_like(inverse_range)
    slow, slow_ttc = collision_probability_from_range(
        inverse_range, base, uncertainty, 0.5, 1.0, 0.08
    )
    fast, fast_ttc = collision_probability_from_range(
        inverse_range, base, uncertainty, 5.0, 1.0, 0.08
    )
    assert np.all(fast > slow)
    assert np.all(fast_ttc < slow_ttc)


def test_nearer_obstacle_is_more_dangerous_for_same_state():
    far, _ = collision_probability_from_range(
        np.full((1, 1), 0.2), np.full((1, 1), 0.1), np.zeros((1, 1)),
        2.0, 1.0, 0.08,
    )
    near, _ = collision_probability_from_range(
        np.full((1, 1), 0.8), np.full((1, 1), 0.1), np.zeros((1, 1)),
        2.0, 1.0, 0.08,
    )
    assert near.item() > far.item()


def test_more_latency_increases_risk_and_reduces_remaining_ttc():
    inverse_range = np.full((1, 1), 0.5, np.float32)
    base = np.full((1, 1), 0.25, np.float32)
    uncertainty = np.zeros((1, 1), np.float32)
    fresh, fresh_ttc = collision_probability_from_range(
        inverse_range, base, uncertainty, 2.0, 1.0, 0.08
    )
    stale, stale_ttc = collision_probability_from_range(
        inverse_range, base, uncertainty, 2.0, 1.0, 0.28
    )
    assert stale.item() > fresh.item()
    assert stale_ttc.item() < fresh_ttc.item()
