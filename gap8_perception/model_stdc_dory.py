"""Resize-free deployment students for the stock DORY NEMO frontend."""

from __future__ import annotations

import copy

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


class Gap8STDCSharedDoryNet(nn.Module):
    """Deployable shared encoder with fine corner and large-RF danger heads."""

    input_shape = (1, 120, 160)
    corner_shape = (4, 30, 40)
    danger_shape = (1, 8, 10)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=2)
        self.corner_head = nn.Sequential(
            DSConv(32, 16), nn.Conv2d(16, 4, 1)
        )
        self.stage2 = ModifiedSTDCStage(32, 64, refinements=3)
        self.stage3 = ModifiedSTDCStage(64, 96, refinements=7)
        self.danger_head = nn.Conv2d(96, 1, 1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(image))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.encode(image)
        danger_features = self.stage3(self.stage2(shared))
        return {
            "corners": self.corner_head(shared),
            "danger": self.danger_head(danger_features),
        }


class Gap8STDCSharedEncoderGraph(nn.Module):
    input_shape = (1, 120, 160)

    def __init__(self, model: Gap8STDCSharedDoryNet | None = None):
        super().__init__()
        model = model or Gap8STDCSharedDoryNet()
        self.stem = copy.deepcopy(model.stem)
        self.stage1 = copy.deepcopy(model.stage1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(image))


class Gap8STDCSharedCornerHeadGraph(nn.Module):
    input_shape = (32, 30, 40)

    def __init__(self, model: Gap8STDCSharedDoryNet | None = None):
        super().__init__()
        model = model or Gap8STDCSharedDoryNet()
        self.head = copy.deepcopy(model.corner_head)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.head(shared)


class Gap8STDCSharedDangerHeadGraph(nn.Module):
    input_shape = (32, 30, 40)

    def __init__(self, model: Gap8STDCSharedDoryNet | None = None):
        super().__init__()
        model = model or Gap8STDCSharedDoryNet()
        self.stage2 = copy.deepcopy(model.stage2)
        self.stage3 = copy.deepcopy(model.stage3)
        self.head = copy.deepcopy(model.danger_head)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage3(self.stage2(shared)))


def shared_deployment_graphs(model: Gap8STDCSharedDoryNet):
    """Split one trained model into three sequential stock-DORY graphs."""
    return (
        Gap8STDCSharedEncoderGraph(model),
        Gap8STDCSharedCornerHeadGraph(model),
        Gap8STDCSharedDangerHeadGraph(model),
    )


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


def initialize_shared_from_rich(
    model: Gap8STDCSharedDoryNet,
    rich_state: dict[str, torch.Tensor],
) -> None:
    """Copy compatible rich encoder and task-head tensors."""
    state = model.state_dict()
    aliases = {
        "corner_head.0.": "corner_fuse40.",
        "corner_head.1.": "corner_logits.",
        "danger_head.": "danger_logits.",
    }
    compatible = {}
    for key, current in state.items():
        source_key = key
        for target_prefix, rich_prefix in aliases.items():
            if key.startswith(target_prefix):
                source_key = rich_prefix + key[len(target_prefix) :]
                break
        if source_key in rich_state and rich_state[source_key].shape == current.shape:
            compatible[key] = rich_state[source_key]
    state.update(compatible)
    model.load_state_dict(state)


def initialize_shared_from_pair(
    model: Gap8STDCSharedDoryNet,
    corner_state: dict[str, torch.Tensor],
    danger_state: dict[str, torch.Tensor],
) -> None:
    """Compose the validated pair while sharing the corner encoder once."""
    state = model.state_dict()
    for key, current in state.items():
        source = None
        if key.startswith(("stem.", "stage1.")):
            source = corner_state.get(key)
        elif key.startswith("corner_head."):
            source = corner_state.get("head." + key[len("corner_head.") :])
        elif key.startswith(("stage2.", "stage3.")):
            source = danger_state.get(key)
        elif key.startswith("danger_head."):
            source = danger_state.get("head." + key[len("danger_head.") :])
        if source is not None and source.shape == current.shape:
            state[key] = source
    model.load_state_dict(state)
