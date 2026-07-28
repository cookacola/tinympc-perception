"""Asymmetric losses for the design-document corner and danger heads."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import soft_dice_loss


def focal_heatmap_mse(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive_gamma: float = 2.0,
    negative_gamma: float = 4.0,
) -> torch.Tensor:
    """CornerNet-style focal weighting applied to Gaussian-heatmap MSE.

    Invalid/no-gate images remain negative supervision.  This is important for
    suppressing phantom gates; only coordinate localization is masked.
    """
    probability = logits.sigmoid()
    squared = (probability - target).square()
    positive_weight = 1.0 + 20.0 * target.pow(positive_gamma)
    negative_weight = (1.0 - target).pow(negative_gamma)
    weights = torch.where(target > 0.05, positive_weight, negative_weight)
    per_image = (squared * weights).flatten(1).mean(1)
    # Valid images and no-gate images both contribute.  The explicit expression
    # documents that ``valid`` controls the target, not wholesale loss masking.
    image_weight = torch.where(valid, 1.0, 0.5).to(per_image.dtype)
    return (per_image * image_weight).sum() / image_weight.sum().clamp_min(1.0)


def design_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    corner_weight: float = 1.0,
    danger_weight: float = 2.0,
    danger_positive_weight: float = 2.5,
) -> dict[str, torch.Tensor]:
    corner = focal_heatmap_mse(
        outputs["corners"], batch["corners"], batch["corner_valid"]
    )
    danger_target = batch["danger"].clamp(0.0, 1.0)
    danger_binary = (danger_target >= 0.5).to(danger_target.dtype)
    danger_bce = F.binary_cross_entropy_with_logits(
        outputs["danger"],
        danger_binary,
        pos_weight=outputs["danger"].new_tensor(danger_positive_weight),
    )
    danger_dice = soft_dice_loss(outputs["danger"], danger_binary)
    danger_regression = F.smooth_l1_loss(
        outputs["danger"].sigmoid(), danger_target
    )
    danger = danger_bce + 0.5 * danger_dice + 0.1 * danger_regression
    return {
        "total": corner_weight * corner + danger_weight * danger,
        "corner": corner,
        "danger": danger,
        "danger_bce": danger_bce,
        "danger_dice": danger_dice,
    }
