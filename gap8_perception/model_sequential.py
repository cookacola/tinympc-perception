"""Deployment-compatible sequential student from the v1 design contract.

The model intentionally has one feed-forward path and one terminal tensor.  It
is the replacement for the former STDC/FPN multi-head model; training-only
teachers and decoders must not be attached to this module before export.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1,
                 groups: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    """A DW 3x3 followed by a PW 1x1, with activation after each convolution."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(
            ConvBNReLU(cin, cin, 3, stride=stride, groups=cin),
            ConvBNReLU(cin, cout, 1),
        )


class SequentialSTDCNet(nn.Module):
    """160x120x1 -> 20x15x12 sequential INT8-oriented CNN.

    Tensor shapes are NCHW.  The terminal projection is deliberately linear;
    all preceding convolutions have bias=False and BN folded at deployment.
    """

    input_shape = (1, 120, 160)
    output_shape = (12, 15, 20)

    def __init__(self, repeated_blocks: int = 6):
        super().__init__()
        if repeated_blocks != 6:
            raise ValueError("the v1 deployment contract requires six DS blocks")
        self.stem = ConvBNReLU(1, 16, 3, stride=2)
        self.layers = nn.Sequential(
            ConvBNReLU(16, 16, 3, groups=16),
            ConvBNReLU(16, 24, 1),
            ConvBNReLU(24, 24, 3, stride=2, groups=24),
            ConvBNReLU(24, 32, 1),
            ConvBNReLU(32, 32, 3, groups=32),
            ConvBNReLU(32, 48, 1),
            ConvBNReLU(48, 48, 3, stride=2, groups=48),
            ConvBNReLU(48, 64, 1),
            ConvBNReLU(64, 64, 3, groups=64),
            ConvBNReLU(64, 96, 1),
            *(DepthwiseSeparableBlock(96, 96) for _ in range(repeated_blocks)),
        )
        self.output = nn.Conv2d(96, 12, 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export() and (
            image.ndim != 4 or tuple(image.shape[1:]) != self.input_shape
        ):
            raise ValueError(
                f"expected NCHW (*,{self.input_shape}), got {tuple(image.shape)}"
            )
        return self.output(self.layers(self.stem(image)))

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(image))


# Stable name for training/export code while migration is in progress.
Gap8SequentialStudent = SequentialSTDCNet
