import numpy as np

from gap8_perception.audit_stdc_real_bias import (
    frame_features,
    temporal_centroid_accuracy,
)


def test_real_bias_features_are_finite():
    image = np.tile(np.arange(160, dtype=np.uint8), (160, 1))
    corners = np.asarray(
        [[40.0, 45.0], [120.0, 45.0], [115.0, 120.0], [45.0, 120.0]],
        np.float32,
    )
    features = frame_features(image, corners)
    assert features.shape == (14,)
    assert np.isfinite(features).all()


def test_temporal_centroid_classifier_detects_clear_shift():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 20)
    features = labels[:, None] * 10.0 + rng.normal(0, 0.1, (60, 2))
    sequence = np.tile(np.arange(5), 12)
    accuracy = temporal_centroid_accuracy(
        features, labels, sequence, slice(None)
    )
    assert accuracy > 0.95
