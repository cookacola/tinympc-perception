import numpy as np
import torch

from gap8_perception.ttc_gate_data import TTCGateDataset, gate_sampling_weights
from gap8_perception.ttc_gate_losses import (
    gate_heatmap_targets,
    gate_perception_loss,
    softargmax_gate_coordinates,
)
from gap8_perception.ttc_motion_gate_model import (
    MotionConditionedESPNetGateTTCNet,
    MotionConditionedESPNetInverseTTCNet,
)
from gap8_perception.ttc_motion_losses import parent_distillation_loss
from gap8_perception.train_ttc_gate_joint_finetune import (
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


def test_invisible_corner_masks_heatmap_and_coordinate_losses():
    prediction = {
        "gate_heatmap_logits": torch.zeros(1, 4, 20, 20),
        "gate_visibility_logits": torch.zeros(1, 4),
    }
    target = {
        "gate_corners_px": torch.tensor([[[20.0, 20.0], [140.0, 20.0],
                                           [140.0, 140.0], [20.0, 140.0]]]),
        "gate_corners_visible": torch.tensor([[True, True, False, False]]),
    }
    _total, parts = gate_perception_loss(prediction, target)
    changed = {key: value.clone() for key, value in prediction.items()}
    changed["gate_heatmap_logits"][:, 2:] = 100.0
    _changed_total, changed_parts = gate_perception_loss(changed, target)
    torch.testing.assert_close(parts["heatmap"], changed_parts["heatmap"])
    torch.testing.assert_close(parts["coordinate"], changed_parts["coordinate"])


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
    for count in range(5):
        valid = np.zeros((3, 4), np.uint8)
        valid[:, :count] = 1
        dataset.trajectories.append({
            "layout_id": f"layout_{count}", "trajectory_id": "trajectory_0",
            "trajectory_type": "test", "targets": {"gate_corners_valid_u8": valid},
        })
        dataset.samples.extend((count, index) for index in range(3))
    weights, summary = gate_sampling_weights(dataset)
    assert np.isclose(weights.sum(), 1.0)
    expected = [0.20, 0.10, 0.30, 0.10, 0.30]
    for count, mass in enumerate(expected):
        selected = slice(count * 3, count * 3 + 3)
        assert np.isclose(weights[selected].sum(), mass)
        assert np.isclose(summary["expected_visible_count_mass"][str(count)], mass)


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
