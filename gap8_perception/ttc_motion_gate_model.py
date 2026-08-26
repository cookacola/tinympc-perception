"""Shared-encoder motion TTC network with a mid-level gate perception branch."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU, DSConv
from .ttc_encoder import ESPNetV2LiteEncoder


class MotionConditionedESPNetInverseTTCNet(nn.Module):
    image_shape = (2, 160, 160)
    onboard_state_shape = (10,)
    output_shape = (1, 20, 20)
    onboard_scale = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0 / 30.0)

    def __init__(self):
        super().__init__()
        self.encoder = ESPNetV2LiteEncoder(input_channels=2)
        self.project_e2 = ConvBNReLU(64, 32, 1)
        self.project_e3 = ConvBNReLU(96, 32, 1)
        self.fuse = DSConv(64, 32)
        self.motion_film = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 64))
        self.inverse_ttc = nn.Conv2d(32, 1, 1)
        self.inverse_depth = nn.Conv2d(32, 1, 1)
        self.flow = nn.Conv2d(32, 2, 1)
        self.risk_logits = nn.Conv2d(32, 3, 1)
        self.register_buffer(
            "onboard_scale_tensor", torch.tensor(self.onboard_scale, dtype=torch.float32),
            persistent=True,
        )
        nn.init.zeros_(self.motion_film[-1].weight)
        nn.init.zeros_(self.motion_film[-1].bias)

    def _validate_inputs(self, images, onboard_state):
        if tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(
                f"expected image shape [B,{self.image_shape}], received {tuple(images.shape)}"
            )
        if tuple(onboard_state.shape[1:]) != self.onboard_state_shape:
            raise ValueError(
                f"expected onboard shape [B,{self.onboard_state_shape}], "
                f"received {tuple(onboard_state.shape)}"
            )

    def _ttc_outputs(self, e2, e3, onboard_state):
        features = self.fuse(torch.cat((
            self.project_e2(e2),
            F.interpolate(self.project_e3(e3), size=e2.shape[-2:], mode="nearest"),
        ), dim=1))
        normalized = torch.clamp(onboard_state / self.onboard_scale_tensor, -4.0, 4.0)
        scale, shift = self.motion_film(normalized).chunk(2, dim=1)
        features = features * (1.0 + 0.5 * torch.tanh(scale[:, :, None, None]))
        features = features + shift[:, :, None, None]
        return {
            "inverse_ttc": F.softplus(self.inverse_ttc(features)),
            "inverse_depth": F.softplus(self.inverse_depth(features)),
            "flow": self.flow(features),
            "risk_logits": self.risk_logits(features),
        }

    def forward(self, images, onboard_state):
        self._validate_inputs(images, onboard_state)
        _e1, e2, e3 = self.encoder(images)
        return self._ttc_outputs(e2, e3, onboard_state)

    def initialize_from_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("model", saved)
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"TTC checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return {
            "checkpoint": str(checkpoint.resolve()),
            "source_epoch": saved.get("epoch"),
            "loaded_tensors": len(state),
            "mode": "full_motion_checkpoint",
            "fresh_tensors": [],
        }


class GateDecoder(nn.Module):
    """Decode four corner heatmaps and four independent visibility logits from e2."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(ConvBNReLU(64, 32, 1), DSConv(32, 32))
        self.corner_heatmap_logits = nn.Conv2d(32, 4, 1)
        self.visibility_logits = nn.Linear(64, 4)

    def forward(self, e2):
        features = self.refine(e2)
        pooled = torch.cat((
            F.adaptive_avg_pool2d(features, 1).flatten(1),
            F.adaptive_max_pool2d(features, 1).flatten(1),
        ), dim=1)
        return {
            "gate_heatmap_logits": self.corner_heatmap_logits(features),
            "gate_visibility_logits": self.visibility_logits(pooled),
        }


class MotionConditionedESPNetGateTTCNet(MotionConditionedESPNetInverseTTCNet):
    """Branch the gate decoder from raw 64x20x20 mid-level ESPNet features."""

    gate_corner_order = ("TL", "TR", "BR", "BL")
    gate_visibility_semantics = "corner_is_in_front_of_camera_and_inside_image"

    def __init__(self):
        super().__init__()
        self.gate_decoder = GateDecoder()

    def forward(self, images, onboard_state):
        self._validate_inputs(images, onboard_state)
        _e1, e2, e3 = self.encoder(images)
        return {**self._ttc_outputs(e2, e3, onboard_state), **self.gate_decoder(e2)}

    def initialize_from_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("model", saved)
        incompatible = self.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        expected_gate = {
            f"gate_decoder.{name}" for name in self.gate_decoder.state_dict()
            if not name.endswith("num_batches_tracked")
        }
        if missing not in (set(), expected_gate) or incompatible.unexpected_keys:
            raise RuntimeError(
                f"gate warm-start mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return {
            "checkpoint": str(checkpoint.resolve()),
            "source_epoch": saved.get("epoch"),
            "loaded_tensors": len(state),
            "mode": "full_gate_checkpoint" if not missing else "v1_ttc_plus_fresh_gate",
            "fresh_tensors": sorted(missing),
        }

    def freeze_parent_for_gate_training(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("gate_decoder."))

    def gate_training_mode(self):
        self.eval()
        self.gate_decoder.train()
        return self
