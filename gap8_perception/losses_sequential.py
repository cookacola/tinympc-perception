"""Losses aligned with the sequential student output contract."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .output_contract import OFFSET_MAX, OFFSET_MIN, SCORE_LIMIT


def decode_offsets(scores: torch.Tensor) -> torch.Tensor:
    clipped = scores.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    return OFFSET_MIN + (clipped + SCORE_LIMIT) * (OFFSET_MAX - OFFSET_MIN) / (2.0 * SCORE_LIMIT)


def sequential_loss(output: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = batch["target"]
    heat_target = target[:, :4]
    heat_weight = 1.0 + 20.0 * ((heat_target + SCORE_LIMIT) / (2.0 * SCORE_LIMIT)).clamp(0, 1).square()
    heatmap = (heat_weight * (output[:, :4] - heat_target).square()).mean()

    offset_scores = output[:, 4:8].mean(dim=(-2, -1))
    predicted_offsets = decode_offsets(offset_scores)
    raw_error = predicted_offsets - batch["offset_m"]
    valid = batch["offset_valid"] * batch.get("offset_loss_mask", 1.0)
    asymmetric = torch.where(raw_error > 0, 2.0, 1.0) * F.smooth_l1_loss(
        predicted_offsets, batch["offset_m"], reduction="none"
    )
    offsets = (asymmetric * valid).sum() / valid.sum().clamp_min(1.0)

    confidence_scores = output[:, 8:12].mean(dim=(-2, -1))
    confidence_target = batch["offset_valid"]
    confidence_mask = batch.get("confidence_loss_mask", torch.ones_like(confidence_target))
    confidence_raw = F.binary_cross_entropy_with_logits(
        confidence_scores, confidence_target, reduction="none"
    )
    confidence = (confidence_raw * confidence_mask).sum() / confidence_mask.sum().clamp_min(1.0)
    field = output[:, 4:12].var(dim=(-2, -1), unbiased=False).mean()
    total = heatmap + 2.0 * offsets + confidence + 0.1 * field
    return {"total": total, "heatmap": heatmap, "offset": offsets, "confidence": confidence, "field": field}
