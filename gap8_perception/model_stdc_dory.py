"""Resize-free deployment students for the stock DORY NEMO frontend."""

from __future__ import annotations

import torch
from torch import nn

from .model import ConvBNReLU, DSConv
from .model_stdc import ModifiedSTDCStage


class Gap8STDCCornerDoryNet(nn.Module):
    """Fine-detail graph: 160x120 mono -> ordered 4x40x30 logits."""

    input_shape = (1, 120, 160)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=2)
        self.head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 4, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage1(self.stem(image)))


class Gap8STDCDangerDoryNet(nn.Module):
    """Large-RF graph: 160x120 mono -> conservative 1x10x8 logits."""

    input_shape = (1, 120, 160)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=2)
        self.stage2 = ModifiedSTDCStage(32, 64, refinements=3)
        self.stage3 = ModifiedSTDCStage(64, 96, refinements=7)
        self.head = nn.Conv2d(96, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stage1(self.stem(image))
        x = self.stage2(x)
        return self.head(self.stage3(x))


def initialize_from_rich(
    corner: Gap8STDCCornerDoryNet,
    danger: Gap8STDCDangerDoryNet,
    rich_state: dict[str, torch.Tensor],
) -> None:
    """Copy every compatible encoder tensor from a rich multi-head checkpoint."""
    for model in (corner, danger):
        state = model.state_dict()
        compatible = {
            key: value
            for key, value in rich_state.items()
            if key in state and state[key].shape == value.shape
        }
        state.update(compatible)
        model.load_state_dict(state)
