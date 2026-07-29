import numpy as np
import pytest

from gap8_perception.audit_real_flights import mocap_speed_summary, topology_valid
from gap8_perception.data_stdc_real import augment_real_gate_frame


def test_real_gate_augmentation_preserves_crop_and_topology():
    image = np.tile(np.arange(160, dtype=np.uint8), (160, 1))
    corners = np.asarray(
        [[45.0, 45.0], [115.0, 43.0], [118.0, 120.0], [42.0, 122.0]],
        np.float32,
    )
    augmented, transformed = augment_real_gate_frame(
        image, corners, np.random.default_rng(2026)
    )
    assert augmented.shape == (160, 160)
    assert augmented.dtype == np.uint8
    assert transformed.shape == (4, 2)
    assert (transformed[:, 0] >= 0).all()
    assert (transformed[:, 0] < 160).all()
    assert (transformed[:, 1] >= 20).all()
    assert (transformed[:, 1] < 140).all()
    assert topology_valid(transformed)


def test_real_gate_augmentation_is_seed_reproducible():
    image = np.full((160, 160), 127, np.uint8)
    corners = np.asarray(
        [[50.0, 50.0], [110.0, 50.0], [110.0, 115.0], [50.0, 115.0]],
        np.float32,
    )
    first = augment_real_gate_frame(
        image, corners, np.random.default_rng(17)
    )
    second = augment_real_gate_frame(
        image, corners, np.random.default_rng(17)
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


def test_mocap_speed_summary_divides_distance_by_time():
    samples = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]]
    )
    summary = mocap_speed_summary(samples)
    assert summary["median_m_per_s"] == pytest.approx(2.0)
    assert summary["fraction_at_or_above_2_m_per_s"] == pytest.approx(1.0)
