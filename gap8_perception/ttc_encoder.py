"""Minimal two-frame ESPNetV2-lite encoder used by the TTC/gate network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU


class EESPBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, branches: int = 4):
        super().__init__()
        if cout % branches:
            raise ValueError("EESP output channels must divide evenly across branches")
        hidden = cout // branches
        self.reduce = ConvBNReLU(cin, hidden, 1)
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    hidden, hidden, 3, stride=stride, padding=dilation,
                    dilation=dilation, groups=hidden, bias=False,
                ),
                nn.BatchNorm2d(hidden),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(cout, cout, 1, groups=branches, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.residual = stride == 1 and cin == cout

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(value)
        outputs, running = [], None
        for path in self.paths:
            current = path(reduced)
            running = current if running is None else running + current
            outputs.append(running)
        encoded = self.project(torch.cat(outputs, 1))
        return F.relu(value + encoded if self.residual else encoded, inplace=True)


class ESPNetV2LiteEncoder(nn.Module):
    """Return 40x40, 20x20 and 10x10 features for 160x160 inputs."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 16, 3, 2)
        self.stage1 = nn.Sequential(EESPBlock(16, 32, 2), EESPBlock(32, 32, 1))
        self.stage2 = nn.Sequential(EESPBlock(32, 64, 2), EESPBlock(64, 64, 1))
        self.stage3 = nn.Sequential(EESPBlock(64, 96, 2), EESPBlock(96, 96, 1))

    def forward(self, value: torch.Tensor):
        value = self.stem(value)
        e1 = self.stage1(value)
        e2 = self.stage2(e1)
        return e1, e2, self.stage3(e2)
