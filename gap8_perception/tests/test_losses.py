import torch

from gap8_perception.losses import multitask_loss


def test_near_perfect_binary_predictions_have_low_loss():
    danger = torch.zeros(1, 1, 20, 20)
    danger[:, :, 4:8, 4:8] = 1
    gate = torch.zeros(1, 1, 40, 40)
    gate[:, :, 10:30, 10:30] = 1
    corners = torch.zeros(1, 4, 40, 40)
    corners[:, :, 20, 20] = 1
    outputs = {
        "danger": torch.where(danger.bool(), 12.0, -12.0),
        "urgency": torch.where(danger.bool(), 12.0, -12.0),
        "uncertainty": torch.full_like(danger, -12.0),
        "gate": torch.where(gate.bool(), 12.0, -12.0),
        "corners": torch.where(corners.bool(), 12.0, -12.0),
    }
    batch = {
        "danger": danger,
        "urgency": danger,
        "uncertainty": torch.zeros_like(danger),
        "gate": gate,
        "corners": corners,
        "corner_valid": torch.tensor([True]),
    }
    losses = multitask_loss(outputs, batch)
    assert losses["danger"] < 1e-3
    assert losses["gate"] < 1e-3
    assert losses["corner"] < 1e-3


def test_invalid_corner_frames_are_fully_masked():
    logits = torch.full((1, 4, 40, 40), -12.0)
    target = torch.zeros_like(logits)
    valid = torch.tensor([False])
    from gap8_perception.losses import weighted_corner_mse

    quiet = weighted_corner_mse(logits, target, valid)
    logits[:, :, 20, 20] = 12.0
    noisy = weighted_corner_mse(logits, target, valid)
    assert torch.equal(noisy, quiet)
    assert quiet == 0
