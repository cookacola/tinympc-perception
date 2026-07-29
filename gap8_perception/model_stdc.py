"""STDC-inspired multi-head network from the 160x120 design document.

The model deliberately keeps the spatial pyramid and two task heads explicit
for training.  Every learned spatial operator is a convolution, depthwise
convolution, or pointwise convolution; nearest-neighbour resize and
concatenation are the only graph plumbing operations.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU, DSConv, ResidualDS


class ModifiedSTDCStage(nn.Sequential):
    """Stride-two DS projection followed by inexpensive residual refinement."""

    def __init__(self, cin: int, cout: int, refinements: int):
        super().__init__(
            DSConv(cin, cout, stride=2),
            *(ResidualDS(cout) for _ in range(refinements)),
        )


class ProposedSTDCFPNNet(nn.Module):
    """Exact 160x120 STDC/FPN proposal with optional training-only heads."""

    input_shape = (1, 120, 160)
    corner_shape = (4, 30, 40)
    danger_shape = (1, 30, 40)

    def __init__(
        self, global_coordinate_head: bool = True, boundary_head: bool = True
    ):
        super().__init__()
        self.stem = ConvBNReLU(1, 16, kernel=3, stride=2)
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=1)
        self.stage2 = ModifiedSTDCStage(32, 64, refinements=1)
        self.stage3 = ModifiedSTDCStage(64, 96, refinements=1)
        self.project3 = ConvBNReLU(96, 16, kernel=1)
        self.project2 = ConvBNReLU(64, 16, kernel=1)
        self.fuse20 = DSConv(32, 16)
        self.project1 = ConvBNReLU(32, 16, kernel=1)
        self.fuse40 = DSConv(16, 16)
        self.corner_logits = nn.Conv2d(16, 4, 1)
        self.danger_logits = nn.Conv2d(16, 1, 1)
        self.boundary_logits = (
            nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
                nn.Conv2d(16, 1, 1),
            )
            if boundary_head
            else None
        )
        self.global_coordinates = (
            nn.Linear(16 * 8 * 10, 8) if global_coordinate_head else None
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="nearest")

    def forward_logits(
        self, image: torch.Tensor, training_heads: bool = False
    ) -> dict[str, torch.Tensor]:
        stem = self.stem(image)
        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        deep = self.project3(stage3)
        fuse20 = self.fuse20(
            torch.cat((self._resize_like(deep, stage2), self.project2(stage2)), 1)
        )
        feature = self.fuse40(
            self._resize_like(fuse20, stage1) + self.project1(stage1)
        )
        outputs = {
            "corners": self.corner_logits(feature),
            "danger": self.danger_logits(feature),
        }
        if training_heads and self.boundary_logits is not None:
            outputs["boundary"] = self.boundary_logits(feature)
        if training_heads and self.global_coordinates is not None:
            outputs["global_coordinates"] = self.global_coordinates(deep.flatten(1))
        return outputs

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_logits(image, training_heads=self.training)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward_logits(image)
        corners = logits["corners"].sigmoid()
        return {
            "corners": corners,
            "corner_confidence": corners.flatten(2).amax(2),
            "danger": logits["danger"].sigmoid(),
        }


class Gap8STDCMultiHeadNet(nn.Module):
    """160x120 mono -> four 40x30 corner maps and one 20x15 danger map."""

    input_shape = (1, 120, 160)
    corner_shape = (4, 30, 40)
    danger_shape = (1, 15, 20)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, 16, kernel=3, stride=2),
            DSConv(16, 16),
        )
        self.stage1 = ModifiedSTDCStage(16, 32, refinements=2)
        self.stage2 = ModifiedSTDCStage(32, 64, refinements=3)
        # Parameters are concentrated at 10x8, where they expand receptive
        # field at low MAC and activation cost.  Seven refinements place the
        # complete network inside the requested 100-180k parameter envelope.
        self.stage3 = ModifiedSTDCStage(64, 96, refinements=7)

        self.corner_fuse20 = DSConv(96 + 64, 32)
        self.corner_fuse40 = DSConv(32 + 32, 16)
        self.corner_logits = nn.Conv2d(16, 4, kernel_size=1)

        self.danger_fuse20 = DSConv(96 + 64, 32)
        self.danger_logits = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="nearest")

    def forward_logits(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)           # 16 x 60 x 80
        stage1 = self.stage1(stem)        # 32 x 30 x 40
        stage2 = self.stage2(stage1)      # 64 x 15 x 20
        stage3 = self.stage3(stage2)      # 96 x  8 x 10

        deep20 = self._resize_like(stage3, stage2)
        corner20 = self.corner_fuse20(torch.cat((deep20, stage2), dim=1))
        corner40 = self.corner_fuse40(
            torch.cat((self._resize_like(corner20, stage1), stage1), dim=1)
        )
        danger20 = self.danger_fuse20(torch.cat((deep20, stage2), dim=1))
        return {
            "corners": self.corner_logits(corner40),
            "danger": self.danger_logits(danger20),
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_logits(image)

    def forward_packed(self, image: torch.Tensor) -> torch.Tensor:
        """Single-output deployment ABI: TL/TR/BR/BL plus replicated danger.

        The receiver conservatively max-pools the last channel back to 20x15.
        Keeping a single tensor mirrors the proven NanoCockpit/DORY interface.
        """
        outputs = self.forward_logits(image)
        danger40 = F.interpolate(
            outputs["danger"], size=outputs["corners"].shape[-2:], mode="nearest"
        )
        return torch.cat((outputs["corners"], danger40), dim=1)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return calibrated domains and one confidence per ordered corner."""
        logits = self.forward_logits(image)
        corners = logits["corners"].sigmoid()
        return {
            "corners": corners,
            "corner_confidence": corners.flatten(2).amax(dim=2),
            "danger": logits["danger"].sigmoid(),
        }


# Stable alias recorded in checkpoints and used by export scripts.
Gap8DesignNet = Gap8STDCMultiHeadNet


class Gap8STDCPrivilegedTeacher(Gap8STDCMultiHeadNet):
    """Training-only teacher that also sees normalized inverse depth."""

    input_shape = (2, 120, 160)

    def __init__(self):
        super().__init__()
        self.stem[0] = ConvBNReLU(2, 16, kernel=3, stride=2)
