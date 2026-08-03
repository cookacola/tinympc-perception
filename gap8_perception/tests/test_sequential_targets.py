import numpy as np
import torch

from gap8_perception.data_sequential import corner_score_fields, encode_offsets
from gap8_perception.hm01b0_sensor import augment_hm01b0
from gap8_perception.losses_sequential import sequential_loss
from gap8_perception.rollout_targets import (
    course_collision_boxes,
    fixed_normal_offsets_to_boxes,
)


def test_fixed_normal_offset_is_first_inflated_box_intersection():
    camera = np.asarray([[100, 0, 80], [0, 100, 60], [0, 0, 1]], np.float32)
    boxes = np.asarray([[[1.0, -0.1, -0.1], [1.2, 0.1, 0.1]]], np.float32)
    offsets, valid = fixed_normal_offsets_to_boxes(
        np.zeros(3), np.asarray([1.0, 0.0, 0.0]), camera,
        np.zeros(5, np.float32), boxes, 0.0, angles_deg=(0.0,),
    )
    np.testing.assert_allclose(offsets, [1.0])
    np.testing.assert_array_equal(valid, [1])


def test_scene_collision_metadata_replaces_legacy_obstacle_layout():
    boxes = course_collision_boxes(
        {"collision_obstacles": [{"center_m": [1.5, 1.0, 0.5], "size_m": [0.2, 0.4, 1.0]}]}
    )
    expected = np.asarray([[1.4, 0.8, 0.0], [1.6, 1.2, 1.0]], np.float32)
    assert any(np.allclose(box, expected) for box in boxes)
    legacy_obstacle = np.asarray([[-0.825, 0.375, 0.0], [-0.275, 0.925, 0.6]], np.float32)
    assert not any(np.allclose(box, legacy_obstacle) for box in boxes)


def test_corner_score_fields_follow_the_20x15_pixel_center_mapping():
    # Grid cell (0, 0) maps to image pixel (3.5, 3.5) after the 20-row crop.
    corners = np.asarray([[3.5, 23.5]] * 4, np.float32)
    fields = corner_score_fields(corners, True)
    assert fields.shape == (4, 15, 20)
    assert np.argmax(fields[0]) == 0
    np.testing.assert_allclose(encode_offsets([0.0, 3.0, 6.0]), [-6.0, 0.0, 6.0])


def test_corner_score_fields_mask_each_corner_independently():
    corners = np.asarray([[3.5, 23.5]] * 4, np.float32)
    fields = corner_score_fields(corners, [True, False, True, False])
    assert fields[0].max() > 0 and fields[2].max() > 0
    assert np.all(fields[1] == -6.0) and np.all(fields[3] == -6.0)


def test_sequential_loss_penalizes_optimistic_offset_more():
    target = torch.zeros(1, 12, 15, 20)
    batch = {
        "target": target,
        "offset_m": torch.full((1, 4), 3.0),
        "offset_valid": torch.ones(1, 4),
    }
    under = target.clone()
    over = target.clone()
    # Scores +/-2 decode to 2 m / 4 m when the target is 3 m.
    under[:, 4:8] = -2.0
    over[:, 4:8] = 2.0
    assert sequential_loss(over, batch)["offset"] > sequential_loss(under, batch)["offset"]


def test_sensor_augmentation_preserves_image_contract_and_can_mark_invalid():
    image = np.full((160, 160), 120, np.uint8)
    augmented, observable = augment_hm01b0(image, np.random.default_rng(4), probability=1.0)
    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint8
    assert isinstance(observable, bool)


def test_real_corner_examples_do_not_invent_safety_supervision():
    target = torch.full((1, 12, 15, 20), -6.0)
    batch = {
        "target": target,
        "offset_m": torch.zeros(1, 4),
        "offset_valid": torch.zeros(1, 4),
        "offset_loss_mask": torch.zeros(1, 4),
        "confidence_loss_mask": torch.zeros(1, 4),
    }
    losses = sequential_loss(target, batch)
    assert losses["offset"] == 0
    assert losses["confidence"] == 0
