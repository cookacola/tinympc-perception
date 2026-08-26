"""Original supervised TTC/depth/flow/risk objective and teacher distillation."""
from __future__ import annotations

import torch
from torch.nn import functional as F


LOSS_WEIGHTS = {"ttc": 0.40, "depth": 0.15, "flow": 0.15, "risk": 0.30}
RISK_CLASS_WEIGHTS = (1.0, 2.0, 6.0)


def risk_class(inverse_ttc):
    return torch.where(
        inverse_ttc >= 2.0,
        torch.full_like(inverse_ttc, 2, dtype=torch.long),
        torch.where(
            inverse_ttc >= (1.0 / 1.5),
            torch.ones_like(inverse_ttc, dtype=torch.long),
            torch.zeros_like(inverse_ttc, dtype=torch.long),
        ),
    )


def _masked_mean(values, mask, weights=None):
    mask = mask.to(values.dtype)
    if weights is not None:
        mask = mask * weights.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def motion_conditioned_ttc_loss(prediction, target):
    approaching = target["ttc_approaching"].bool()
    valid = target["ttc_valid"].bool()
    positive = (approaching & valid).sum().to(torch.float32)
    negative = ((~approaching) & valid).sum().to(torch.float32)
    positive_weight = torch.clamp(negative / positive.clamp_min(1.0), 1.0, 12.0)
    ttc_weights = torch.where(approaching, positive_weight, 1.0)
    ttc = _masked_mean(
        F.smooth_l1_loss(prediction["inverse_ttc"], target["inverse_ttc"], reduction="none"),
        valid,
        ttc_weights,
    )
    depth = _masked_mean(
        F.smooth_l1_loss(prediction["inverse_depth"], target["inverse_depth"], reduction="none"),
        target["depth_valid"],
    )
    flow = _masked_mean(
        torch.sqrt(((prediction["flow"] - target["flow"]) ** 2).sum(1, keepdim=True) + 1e-6),
        target["flow_valid"],
    )
    labels = risk_class(target["inverse_ttc"])
    labels = torch.where(approaching, labels, torch.zeros_like(labels)).squeeze(1)
    logits = prediction["risk_logits"]
    class_weights = logits.new_tensor(RISK_CLASS_WEIGHTS)
    cross_entropy = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    probability = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    focal = (1.0 - probability).square() * cross_entropy
    risk = _masked_mean(focal, valid.squeeze(1))
    parts = {"ttc": ttc, "depth": depth, "flow": flow, "risk": risk}
    return sum(LOSS_WEIGHTS[name] * value for name, value in parts.items()), parts


def parent_distillation_loss(student, teacher, temperature=2.0):
    regression = {
        "ttc": F.smooth_l1_loss(student["inverse_ttc"], teacher["inverse_ttc"]),
        "depth": F.smooth_l1_loss(student["inverse_depth"], teacher["inverse_depth"]),
        "flow": F.smooth_l1_loss(student["flow"], teacher["flow"]),
    }
    teacher_probability = torch.softmax(teacher["risk_logits"] / temperature, dim=1)
    risk = F.kl_div(
        torch.log_softmax(student["risk_logits"] / temperature, dim=1),
        teacher_probability,
        reduction="batchmean",
    ) * (temperature ** 2) / student["risk_logits"].shape[-2:].numel()
    parts = {**regression, "risk": risk}
    return sum(LOSS_WEIGHTS[name] * value for name, value in parts.items()), parts
