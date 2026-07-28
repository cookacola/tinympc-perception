"""Multi-task losses emphasizing false-safe danger errors."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def weighted_corner_mse(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    prediction = logits.sigmoid()
    loss = logits.sum() * 0.0
    if valid.any():
        truth = target[valid]
        weights = 1.0 + 20.0 * truth
        loss = ((prediction[valid] - truth).square() * weights).mean()
    # Invalid includes distant, edge-on, partial, and occluded gates. There is
    # no trustworthy corner error on those frames, so they are fully masked.
    # Gate visibility/confidence remains supervised by the opening-mask head.
    return loss


def multitask_loss(outputs, batch, gate_weight: float = 0.5):
    corner = weighted_corner_mse(
        outputs["corners"], batch["corners"], batch["corner_valid"]
    )
    danger_binary = (batch["danger"] >= 0.5).to(batch["danger"].dtype)
    danger_bce = F.binary_cross_entropy_with_logits(
        outputs["danger"],
        danger_binary,
        # Mildly conservative weighting; rollout labels are already majority
        # positive, so the old 4x bootstrap weighting caused saturation.
        pos_weight=outputs["danger"].new_tensor(1.5),
    )
    danger_risk_regression = F.smooth_l1_loss(
        outputs["danger"].sigmoid(), batch["danger"]
    )
    danger = (
        danger_bce
        + 0.5 * soft_dice_loss(outputs["danger"], danger_binary)
        + 0.1 * danger_risk_regression
    )
    urgency = F.smooth_l1_loss(outputs["urgency"].sigmoid(), batch["urgency"])
    uncertainty = F.smooth_l1_loss(
        outputs["uncertainty"].sigmoid(), batch["uncertainty"]
    )
    gate = outputs["danger"].sum() * 0.0
    if "gate" in outputs:
        gate = F.binary_cross_entropy_with_logits(outputs["gate"], batch["gate"])
        gate = gate + 0.5 * soft_dice_loss(outputs["gate"], batch["gate"])
    return {
        "total": (
            corner
            + danger
            + 0.25 * urgency
            + 0.10 * uncertainty
            + gate_weight * gate
        ),
        "corner": corner,
        "danger": danger,
        "urgency": urgency,
        "uncertainty": uncertainty,
        "gate": gate,
    }
