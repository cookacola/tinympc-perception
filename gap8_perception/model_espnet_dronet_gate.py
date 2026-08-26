"""Deployment-oriented two-frame ESPNet with DroNet navigation and gate heads."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU, DSConv
from .model_espnet_dory_student import ESPNetDoryStudent


class ESPNetDroNetGate(nn.Module):
    """Two frames -> DroNet scalars plus middle-tap gate evidence.

    The model deliberately exposes logits. Sigmoid and structured gate
    confidence fusion are controller-side operations, as in PULP-DroNet.
    """

    input_shape = (2, 160, 160)
    middle_shape = (64, 20, 20)
    corner_shape = (4, 20, 20)
    gate_shape = (1, 20, 20)
    navigation_shape = (2,)

    def __init__(self):
        super().__init__()
        source = ESPNetDoryStudent()
        self.stem = source.stem
        self.stage1 = source.stage1
        self.stage2 = source.stage2
        self.stage3 = source.stage3

        # The original ESPNet tap ablation selected stage 2 for both tasks.
        self.corner_adapter = ConvBNReLU(64, 32, 1)
        self.corner_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 4, 1))
        self.gate_adapter = ConvBNReLU(64, 32, 1)
        self.gate_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 1, 1))
        self.presence_pool = nn.AdaptiveAvgPool2d(1)
        self.presence_head = nn.Linear(64, 1)

        self.navigation_pool = nn.AdaptiveAvgPool2d(1)
        self.navigation_dropout = nn.Dropout(p=0.5)
        self.navigation_head = nn.Linear(96, 2, bias=False)

    def middle_features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(self.stem(frames)))

    def forward_gate_raw(self, middle: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "corners_raw": self.corner_head(self.corner_adapter(middle)),
            "gate_raw": self.gate_head(self.gate_adapter(middle)),
            "presence_logit": self.presence_head(
                self.presence_pool(middle).flatten(1)
            ).squeeze(1),
        }

    def forward_navigation_raw(self, middle: torch.Tensor) -> torch.Tensor:
        deep = self.stage3(middle)
        return self.navigation_head(
            self.navigation_dropout(self.navigation_pool(deep).flatten(1))
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        middle = self.middle_features(frames)
        output = self.forward_gate_raw(middle)
        output["navigation_logits"] = self.forward_navigation_raw(middle)
        return output

    def navigation(self, frames: torch.Tensor):
        logits = self.forward_navigation_raw(self.middle_features(frames))
        return [logits[:, 0], torch.sigmoid(logits[:, 1])]

    def gate(self, frames: torch.Tensor, output_size=(40, 40)):
        raw = self.forward_gate_raw(self.middle_features(frames))
        return {
            "corners": F.interpolate(
                raw["corners_raw"], output_size, mode="bilinear", align_corners=False
            ),
            "gate": F.interpolate(
                raw["gate_raw"], output_size, mode="bilinear", align_corners=False
            ),
            "presence_logit": raw["presence_logit"],
        }


class SharedEncoderGraph(nn.Module):
    input_shape = ESPNetDroNetGate.input_shape

    def __init__(self, model: ESPNetDroNetGate):
        super().__init__()
        self.stem = copy.deepcopy(model.stem)
        self.stage1 = copy.deepcopy(model.stage1)
        self.stage2 = copy.deepcopy(model.stage2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(self.stem(frames)))


class SpatialGateGraph(nn.Module):
    input_shape = ESPNetDroNetGate.middle_shape

    def __init__(self, adapter: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = copy.deepcopy(adapter)
        self.head = copy.deepcopy(head)

    def forward(self, middle: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(middle))


class PresenceGraph(nn.Module):
    input_shape = ESPNetDroNetGate.middle_shape

    def __init__(self, model: ESPNetDroNetGate):
        super().__init__()
        self.pool = copy.deepcopy(model.presence_pool)
        self.head = copy.deepcopy(model.presence_head)

    def forward(self, middle: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(middle).flatten(1))


class NavigationGraph(nn.Module):
    input_shape = ESPNetDroNetGate.middle_shape

    def __init__(self, model: ESPNetDroNetGate):
        super().__init__()
        self.stage3 = copy.deepcopy(model.stage3)
        self.pool = copy.deepcopy(model.navigation_pool)
        self.head = copy.deepcopy(model.navigation_head)

    def forward(self, middle: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stage3(middle)).flatten(1))


def deployment_graphs(model: ESPNetDroNetGate):
    return {
        "encoder": SharedEncoderGraph(model),
        "corner_head": SpatialGateGraph(model.corner_adapter, model.corner_head),
        "gate_head": SpatialGateGraph(model.gate_adapter, model.gate_head),
        "presence_head": PresenceGraph(model),
        "navigation_head": NavigationGraph(model),
    }
