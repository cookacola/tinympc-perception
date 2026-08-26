"""Two-frame, resize-free NEMO/DORY student for the ESPNet teacher."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .model import ConvBNReLU, DSConv, ResidualDS


class ModifiedSTDCStage(nn.Sequential):
    def __init__(self, cin: int, cout: int, refinements: int):
        super().__init__(
            DSConv(cin, cout, stride=2),
            *(ResidualDS(cout) for _ in range(refinements)),
        )


class DorySequentialStage(nn.Sequential):
    """DORY-safe downsampling followed by strictly sequential refinements."""

    def __init__(self, cin: int, cout: int, refinements: int):
        super().__init__(
            ConvBNReLU(cin, cout, kernel=3, stride=2),
            *(DSConv(cout, cout) for _ in range(refinements)),
        )


class ESPNetDoryStudent(nn.Module):
    """2x160x160 frames -> corners, gate mask, and 10x10 danger logits."""

    input_shape = (2, 160, 160)
    shared_shape = (32, 40, 40)
    corner_shape = (4, 40, 40)
    gate_shape = (1, 40, 40)
    danger_shape = (1, 10, 10)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(2, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=2)
        self.corner_head = nn.Sequential(
            DSConv(32, 16), nn.Conv2d(16, 4, 1)
        )
        self.gate_head = nn.Sequential(
            DSConv(32, 16), nn.Conv2d(16, 1, 1)
        )
        self.stage2 = ModifiedSTDCStage(32, 64, refinements=3)
        self.stage3 = ModifiedSTDCStage(64, 96, refinements=7)
        self.danger_head = nn.Conv2d(96, 1, 1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(frames))

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.encode(frames)
        deep = self.stage3(self.stage2(shared))
        return {
            "corners": self.corner_head(shared),
            "gate": self.gate_head(shared),
            "danger": self.danger_head(deep),
        }


class DorySequentialStudent(nn.Module):
    """Residual-free student restricted to reliable GAP8/DORY operators."""

    input_shape = ESPNetDoryStudent.input_shape
    shared_shape = ESPNetDoryStudent.shared_shape
    corner_shape = ESPNetDoryStudent.corner_shape
    gate_shape = ESPNetDoryStudent.gate_shape
    danger_shape = ESPNetDoryStudent.danger_shape

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(2, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = DorySequentialStage(16, 32, refinements=2)
        self.corner_head = nn.Sequential(
            DSConv(32, 16), nn.Conv2d(16, 4, 1)
        )
        self.gate_head = nn.Sequential(
            DSConv(32, 16), nn.Conv2d(16, 1, 1)
        )
        self.stage2 = DorySequentialStage(32, 64, refinements=2)
        self.stage3 = DorySequentialStage(64, 96, refinements=4)
        self.danger_head = nn.Conv2d(96, 1, 1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(frames))

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.encode(frames)
        deep = self.stage3(self.stage2(shared))
        return {
            "corners": self.corner_head(shared),
            "gate": self.gate_head(shared),
            "danger": self.danger_head(deep),
        }


class EncoderGraph(nn.Module):
    input_shape = ESPNetDoryStudent.input_shape

    def __init__(self, model: ESPNetDoryStudent):
        super().__init__()
        self.stem = copy.deepcopy(model.stem)
        self.stage1 = copy.deepcopy(model.stage1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(frames))


class FineHeadGraph(nn.Module):
    input_shape = ESPNetDoryStudent.shared_shape

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = copy.deepcopy(head)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.head(shared)


class DangerHeadGraph(nn.Module):
    input_shape = ESPNetDoryStudent.shared_shape

    def __init__(self, model: ESPNetDoryStudent):
        super().__init__()
        self.stage2 = copy.deepcopy(model.stage2)
        self.stage3 = copy.deepcopy(model.stage3)
        self.head = copy.deepcopy(model.danger_head)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage3(self.stage2(shared)))


def build_student(architecture: str):
    if architecture == "ESPNetDoryStudent":
        return ESPNetDoryStudent()
    if architecture == "DorySequentialStudent":
        return DorySequentialStudent()
    raise ValueError(f"unsupported student architecture: {architecture}")


def deployment_graphs(model: ESPNetDoryStudent | DorySequentialStudent):
    return {
        "encoder": EncoderGraph(model),
        "corner_head": FineHeadGraph(model.corner_head),
        "gate_head": FineHeadGraph(model.gate_head),
        "danger_head": DangerHeadGraph(model),
    }
