import numpy as np
import torch

from gap8_perception.ttc_gate_data import TTCGateDataset, gate_sampling_weights
from gap8_perception.ttc_gate_losses import (
    gate_heatmap_targets,
    gate_perception_loss,
    peak_gate_coordinates,
    softargmax_gate_coordinates,
)
from gap8_perception.ttc_motion_gate_model import (
    MotionConditionedESPNetGateTTCNet,
    MotionConditionedESPNetInverseTTCNet,
)
from gap8_perception.audit_ttc_gate_dory_graphs import audit_graphs
from gap8_perception.ttc_motion_gate_dory_model import (
    DoryPartitionedMotionGateTTCNet,
)
from gap8_perception.ttc_motion_losses import parent_distillation_loss
from gap8_perception.train_ttc_gate_joint_finetune import (
    TEST_LIMITS,
    configure_trainable,
    retention_passes,
)


def test_gate_model_warm_starts_v1_and_preserves_parent_outputs(tmp_path):
    baseline = MotionConditionedESPNetInverseTTCNet().eval()
    checkpoint = tmp_path / "v1.pt"
    torch.save({"epoch": 20, "model": baseline.state_dict()}, checkpoint)
    joint = MotionConditionedESPNetGateTTCNet().eval()
    report = joint.initialize_from_checkpoint(checkpoint)
    assert report["mode"] == "v1_ttc_plus_fresh_gate"
    assert report["source_epoch"] == 20
    assert all(name.startswith("gate_decoder.") for name in report["fresh_tensors"])
    images = torch.randn(2, 2, 160, 160)
    onboard = torch.randn(2, 10)
    with torch.no_grad():
        original = baseline(images, onboard)
        augmented = joint(images, onboard)
    for name in original:
        torch.testing.assert_close(original[name], augmented[name], rtol=0, atol=0)
    assert augmented["gate_heatmap_logits"].shape == (2, 4, 20, 20)
    assert augmented["gate_visibility_logits"].shape == (2, 4)


def test_gate_only_optimization_leaves_parent_parameters_and_buffers_exact():
    model = MotionConditionedESPNetGateTTCNet()
    model.freeze_parent_for_gate_training()
    before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("gate_decoder.")
    }
    model.gate_training_mode()
    target = {
        "gate_corners_px": torch.tensor([[[20.0, 20.0], [140.0, 20.0],
                                           [140.0, 140.0], [20.0, 140.0]]]),
        "gate_corners_visible": torch.ones(1, 4, dtype=torch.bool),
        "gate_supervision_eligible": torch.ones(1, dtype=torch.bool),
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    prediction = model(torch.randn(1, 2, 160, 160), torch.randn(1, 10))
    loss, _parts = gate_perception_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("gate_decoder.")
    )


def test_heatmap_targets_follow_tl_tr_br_bl_order_and_decode_to_cell_centers():
    corners = torch.tensor([[[3.5, 3.5], [155.5, 3.5], [155.5, 155.5], [3.5, 155.5]]])
    visible = torch.ones(1, 4, dtype=torch.bool)
    heatmaps = gate_heatmap_targets(corners, visible)
    peaks = heatmaps.flatten(2).argmax(-1)
    assert peaks.tolist() == [[0, 19, 399, 380]]
    logits = torch.full((1, 4, 20, 20), -20.0)
    for corner, index in enumerate(peaks[0]):
        logits[0, corner].view(-1)[index] = 20.0
    decoded = softargmax_gate_coordinates(logits)
    torch.testing.assert_close(decoded, corners, atol=1e-4, rtol=0)
    torch.testing.assert_close(peak_gate_coordinates(logits), corners, atol=0, rtol=0)


def test_invisible_corner_masks_heatmap_and_coordinate_losses():
    prediction = {
        "gate_heatmap_logits": torch.zeros(1, 4, 20, 20),
        "gate_visibility_logits": torch.zeros(1, 4),
    }
    target = {
        "gate_corners_px": torch.tensor([[[20.0, 20.0], [140.0, 20.0],
                                           [140.0, 140.0], [20.0, 140.0]]]),
        "gate_corners_visible": torch.tensor([[True, True, False, False]]),
        "gate_supervision_eligible": torch.ones(1, dtype=torch.bool),
    }
    _total, parts = gate_perception_loss(prediction, target)
    changed = {key: value.clone() for key, value in prediction.items()}
    changed["gate_heatmap_logits"][:, 2:] = 100.0
    _changed_total, changed_parts = gate_perception_loss(changed, target)
    torch.testing.assert_close(parts["heatmap"], changed_parts["heatmap"])
    torch.testing.assert_close(parts["coordinate"], changed_parts["coordinate"])


def test_spatial_heatmap_loss_prefers_the_correct_corner_peak():
    target = {
        "gate_corners_px": torch.tensor([[[3.5, 3.5], [155.5, 3.5],
                                           [155.5, 155.5], [3.5, 155.5]]]),
        "gate_corners_visible": torch.ones(1, 4, dtype=torch.bool),
        "gate_supervision_eligible": torch.ones(1, dtype=torch.bool),
    }
    correct_logits = torch.full((1, 4, 20, 20), -4.0)
    wrong_logits = torch.full((1, 4, 20, 20), -4.0)
    correct_peaks = (0, 19, 399, 380)
    wrong_peaks = (399, 380, 0, 19)
    for corner in range(4):
        correct_logits[0, corner].view(-1)[correct_peaks[corner]] = 4.0
        wrong_logits[0, corner].view(-1)[wrong_peaks[corner]] = 4.0
    visibility_logits = torch.full((1, 4), 4.0)
    _correct_total, correct = gate_perception_loss({
        "gate_heatmap_logits": correct_logits,
        "gate_visibility_logits": visibility_logits,
    }, target)
    _wrong_total, wrong = gate_perception_loss({
        "gate_heatmap_logits": wrong_logits,
        "gate_visibility_logits": visibility_logits,
    }, target)
    assert correct["heatmap"] < wrong["heatmap"]


def test_horizontal_flip_preserves_semantic_corner_order(monkeypatch):
    sample = {
        "images": np.zeros((2, 160, 160), np.float32),
        "inverse_ttc": np.zeros((1, 20, 20), np.float32),
        "ttc_valid": np.ones((1, 20, 20), bool),
        "ttc_approaching": np.ones((1, 20, 20), bool),
        "inverse_depth": np.zeros((1, 20, 20), np.float32),
        "depth_valid": np.ones((1, 20, 20), bool),
        "flow": np.zeros((2, 20, 20), np.float32),
        "flow_valid": np.ones((1, 20, 20), bool),
        "onboard_state": np.zeros(10, np.float32),
        "gate_corners_px": np.asarray([[10, 20], [140, 20], [140, 130], [10, 130]], np.float32),
        "gate_corners_visible": np.asarray([True, False, True, False]),
    }
    monkeypatch.setattr(np.random, "uniform", lambda *_args: 1.0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    TTCGateDataset._augment(sample)
    np.testing.assert_allclose(
        sample["gate_corners_px"], [[19, 20], [149, 20], [149, 130], [19, 130]]
    )
    assert sample["gate_corners_visible"].tolist() == [False, True, False, True]


def test_gate_sampler_reproduces_predeclared_visible_count_masses():
    class FakeDataset:
        trajectories = []
        samples = []

        def __len__(self):
            return len(self.samples)

    dataset = FakeDataset()
    dataset.gate_supervision_policy = lambda: {"test": True}
    for count in range(5):
        valid = np.zeros((3, 4), np.uint8)
        valid[:, :count] = 1
        dataset.trajectories.append({
            "layout_id": f"layout_{count}", "trajectory_id": "trajectory_0",
            "trajectory_type": "test", "targets": {"gate_corners_valid_u8": valid},
            "gate_geometry": {"eligible": np.ones(3, dtype=bool)},
        })
        dataset.samples.extend((count, index) for index in range(3))
    weights, summary = gate_sampling_weights(dataset)
    assert np.isclose(weights.sum(), 1.0)
    expected = [0.20, 0.10, 0.30, 0.10, 0.30]
    for count, mass in enumerate(expected):
        selected = slice(count * 3, count * 3 + 3)
        assert np.isclose(weights[selected].sum(), mass)
        assert np.isclose(summary["expected_visible_count_mass"][str(count)], mass)


def test_gate_quality_mask_keeps_large_partial_and_true_negative_but_excludes_far_small():
    dataset = TTCGateDataset.__new__(TTCGateDataset)
    dataset.maximum_gate_distance_m = 8.0
    dataset.minimum_gate_span_px = 16.0
    dataset.minimum_gate_area_px2 = 256.0
    frames = [
        {"vehicle_state": {"position_m": [distance, 0.0, 0.0]}}
        for distance in (0.0, 0.0, -9.0, 0.0)
    ]
    large = np.asarray([[20, 20], [40, 20], [40, 40], [20, 40]], np.float32)
    small = np.asarray([[20, 20], [30, 20], [30, 30], [20, 30]], np.float32)
    targets = {
        "gate_index_i16": np.asarray([-1, 0, 0, 0], np.int16),
        "gate_corners_px_f32": np.stack((large, large, large, small)),
        "gate_corners_valid_u8": np.asarray([
            [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]
        ], np.uint8),
    }
    geometry = dataset._gate_geometry(frames, targets, [{"center_m": [0, 0, 0]}])
    assert geometry["eligible"].tolist() == [True, True, False, False]


def test_ineligible_gate_sample_contributes_zero_loss_and_gradient():
    logits = torch.randn(1, 4, 20, 20, requires_grad=True)
    visibility = torch.randn(1, 4, requires_grad=True)
    target = {
        "gate_corners_px": torch.zeros(1, 4, 2),
        "gate_corners_visible": torch.ones(1, 4, dtype=torch.bool),
        "gate_supervision_eligible": torch.zeros(1, dtype=torch.bool),
    }
    loss, parts = gate_perception_loss({
        "gate_heatmap_logits": logits, "gate_visibility_logits": visibility,
    }, target)
    assert loss == 0
    assert all(value == 0 for value in parts.values())
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert torch.count_nonzero(visibility.grad) == 0


def test_parent_distillation_is_zero_for_identical_outputs():
    output = {
        "inverse_ttc": torch.randn(2, 1, 20, 20),
        "inverse_depth": torch.randn(2, 1, 20, 20),
        "flow": torch.randn(2, 2, 20, 20),
        "risk_logits": torch.randn(2, 3, 20, 20),
    }
    total, parts = parent_distillation_loss(output, output)
    assert total.abs() < 1e-6
    assert all(value.abs() < 1e-6 for value in parts.values())


def test_joint_phase_unfreezes_only_gate_and_last_e2_block():
    model = MotionConditionedESPNetGateTTCNet()
    gate, encoder = configure_trainable(model)
    assert sum(parameter.numel() for parameter in gate) == 3944
    assert sum(parameter.numel() for parameter in encoder) > 0
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(
        name.startswith("gate_decoder.") or name.startswith("encoder.stage2.1.")
        for name in trainable_names
    )
    assert any(name.startswith("encoder.stage2.1.") for name in trainable_names)


def test_all_mid_scope_unfreezes_stem_through_e2_but_not_deeper_ttc_path():
    model = MotionConditionedESPNetGateTTCNet()
    _gate, encoder = configure_trainable(model, "all_mid")
    assert sum(parameter.numel() for parameter in encoder) > 2912
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("encoder.stem.") for name in trainable_names)
    assert any(name.startswith("encoder.stage1.") for name in trainable_names)
    assert any(name.startswith("encoder.stage2.") for name in trainable_names)
    assert not any(name.startswith("encoder.stage3.") for name in trainable_names)
    assert not any(name.startswith("project_e2.") for name in trainable_names)


def test_retention_limits_require_all_parent_metrics():
    passing = {
        "inverse_ttc_mae_s_inv": 0.16,
        "approaching_inverse_ttc_mae_s_inv": 0.18,
        "inverse_depth_mae_m_inv": 0.23,
        "flow_epe_cells": 0.13,
        "critical_precision_at_0_552": 0.70,
        "critical_recall_at_0_552": 0.74,
    }
    assert retention_passes(passing)
    for name in passing:
        failing = dict(passing)
        failing[name] = 0.0 if name.startswith("critical_") else 1.0
        assert not retention_passes(failing)
    test_passing = {
        "inverse_ttc_mae_s_inv": 0.1502,
        "approaching_inverse_ttc_mae_s_inv": 0.1660,
        "inverse_depth_mae_m_inv": 0.1980,
        "flow_epe_cells": 0.1220,
        "critical_precision_at_0_552": 0.6820,
        "critical_recall_at_0_552": 0.7410,
    }
    assert retention_passes(test_passing, TEST_LIMITS)


def test_dory_partition_shapes_and_state_plane_packing():
    model = DoryPartitionedMotionGateTTCNet().eval()
    images = torch.randn(2, 2, 160, 160)
    onboard = torch.tensor([
        [3, -3, 0, 6, -6, 0, 1, -1, 0.5, 1 / 30],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    with torch.no_grad():
        e2 = model.encoder(images)
        packed = model.pack_ttc_input(e2, onboard)
        output = model(images, onboard)
    assert e2.shape == (2, 64, 20, 20)
    assert packed.shape == (2, 74, 20, 20)
    torch.testing.assert_close(packed[:, :64], e2)
    assert torch.all(packed[0, 64] == 1.0)
    assert torch.all(packed[0, 67] == 2.0)
    assert torch.all(packed[0, 68] == -2.0)
    assert torch.all(packed[0, 73] == 1.0)
    assert output["gate_heatmap_logits"].shape == (2, 4, 20, 20)
    assert output["gate_visibility_logits"].shape == (2, 4)
    assert output["inverse_ttc"].shape == (2, 1, 20, 20)


def test_all_three_deployment_graphs_use_only_stock_dory_operators(tmp_path):
    report = audit_graphs(DoryPartitionedMotionGateTTCNet().eval(), tmp_path)
    assert report["passed"]
    assert set(report["graphs"]) == {"encoder", "gate_head", "ttc_head"}
    assert all(graph["inputs"] == 1 for graph in report["graphs"].values())
