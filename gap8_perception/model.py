"""Single-wire DORY/GAP8 multi-task perception network."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                cin,
                cout,
                kernel,
                stride=stride,
                padding=kernel // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, both foldable at inference."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNReLU(cin, cin, 3, stride, groups=cin)
        self.pointwise = ConvBNReLU(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualDS(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = DSConv(channels, channels)
        self.add = ElementwiseAdd()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.add(x, self.block(x)))


class ElementwiseAdd(nn.Module):
    """Named residual add replaced by PACT_IntegerAdd in the NEMO mirror."""

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return first + second


class Gap8MultiTaskNet(nn.Module):
    """160x160 mono -> corners 4x40x40, danger 1x20x20, gate 1x40x40."""

    def __init__(self, gate_head: bool = True, state_dim: int = 8):
        super().__init__()
        self.gate_head_enabled = gate_head
        self.state_dim = state_dim

        self.stem = nn.Sequential(ConvBNReLU(1, 8, 3, 2), DSConv(8, 12))
        self.e1_down = DSConv(12, 20, 2)
        self.e1_refine = ResidualDS(20)
        self.geometry40 = nn.Sequential(
            ConvBNReLU(20, 16, 1),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
            ResidualDS(16),
        )
        self.packed_channels = 8 if gate_head else 7
        self.packed_head = nn.Sequential(
            DSConv(16, 12), nn.Conv2d(12, self.packed_channels, 1)
        )

    def forward_packed(self, x: torch.Tensor) -> torch.Tensor:
        """Raw single-output graph exported to NEMO/DORY."""
        x = self.stem(x)                       # 80
        e1 = self.e1_refine(self.e1_down(x))   # 40
        return self.packed_head(self.geometry40(e1))

    def forward_image(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single-input inference path consumed by the stock DORY frontend."""
        packed = self.forward_packed(x)
        outputs = {
            "corners": packed[:, 0:4],
            "danger": torch.nn.functional.avg_pool2d(
                packed[:, 4:5], kernel_size=2, stride=2
            ),
            "urgency": torch.nn.functional.avg_pool2d(
                packed[:, 5:6], kernel_size=2, stride=2
            ),
            "uncertainty": torch.nn.functional.avg_pool2d(
                packed[:, 6:7], kernel_size=2, stride=2
            ),
        }
        if self.gate_head_enabled:
            outputs["gate"] = packed[:, 7:8]
        return outputs

    def forward(
        self, x: torch.Tensor, vehicle_state: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # vehicle_state remains in the training API for metadata compatibility,
        # but the DORY CNN is intentionally image-only. Speed-dependent danger
        # is computed in the controller-facing postprocessor.
        return self.forward_image(x)
