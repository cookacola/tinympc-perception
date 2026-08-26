"""Stock-DORY-partitioned two-frame gate and motion-TTC network."""
from __future__ import annotations

from pathlib import Path
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU, DSConv, ElementwiseAdd, ResidualDS


class DoryStage(nn.Sequential):
    """Resize-free depthwise stage with only Conv/ReLU/residual-Add operators."""

    def __init__(self, cin: int, cout: int, refinements: int, stride: int = 2):
        super().__init__(
            DSConv(cin, cout, stride),
            *(ResidualDS(cout) for _ in range(refinements)),
        )


class DoryTwoFrameEncoder(nn.Module):
    input_shape = (2, 160, 160)
    output_shape = (64, 20, 20)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(2, 16, 3, 2), DSConv(16, 16))
        self.stage1 = DoryStage(16, 32, refinements=2)
        self.stage2 = DoryStage(32, 64, refinements=3)

    def forward(self, images):
        return self.stage2(self.stage1(self.stem(images)))


class DoryGateHead(nn.Module):
    input_shape = (64, 20, 20)
    output_shape = (8, 20, 20)
    corner_channels = slice(0, 4)
    visibility_channels = slice(4, 8)

    def __init__(self):
        super().__init__()
        self.adapter = ConvBNReLU(64, 32, 1)
        self.head_features = DSConv(32, 16)
        self.output = nn.Conv2d(16, 8, 1)

    def forward(self, e2):
        return self.output(self.head_features(self.adapter(e2)))


class DoryMotionTTCHead(nn.Module):
    """One-input graph over host-packed e2 and normalized state planes."""

    input_shape = (74, 20, 20)
    output_shape = (7, 20, 20)

    def __init__(self, refinements: int = 3):
        super().__init__()
        self.refinements = int(refinements)
        self.adapter = ConvBNReLU(74, 64, 1)
        self.deep = nn.Sequential(
            *(ResidualDS(64) for _ in range(self.refinements)),
            DSConv(64, 32),
        )
        self.shortcut = ConvBNReLU(74, 32, 1)
        self.add = ElementwiseAdd()
        self.relu = nn.ReLU(inplace=False)
        self.output = nn.Conv2d(32, 7, 1)

    def forward(self, packed_e2_and_state):
        deep = self.deep(self.adapter(packed_e2_and_state))
        return self.output(self.relu(self.add(deep, self.shortcut(packed_e2_and_state))))


class DoryPartitionedMotionGateTTCNet(nn.Module):
    """Training wrapper around three independently compilable DORY graphs."""

    image_shape = (2, 160, 160)
    onboard_state_shape = (10,)
    onboard_scale = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0 / 30.0)
    gate_corner_order = ("TL", "TR", "BR", "BL")
    gate_visibility_semantics = "corner_is_in_front_of_camera_and_inside_image"
    deployment_graphs = ("encoder", "gate_head", "ttc_head")

    def __init__(self, ttc_refinements: int = 3):
        super().__init__()
        self.ttc_refinements = int(ttc_refinements)
        self.encoder = DoryTwoFrameEncoder()
        self.gate_head = DoryGateHead()
        self.ttc_head = DoryMotionTTCHead(self.ttc_refinements)
        self.register_buffer(
            "onboard_scale_tensor",
            torch.tensor(self.onboard_scale, dtype=torch.float32),
            persistent=True,
        )

    def normalized_state_planes(self, onboard_state, height=20, width=20):
        normalized = torch.clamp(onboard_state / self.onboard_scale_tensor, -4.0, 4.0)
        return normalized[:, :, None, None].expand(-1, -1, height, width)

    def pack_ttc_input(self, e2, onboard_state):
        state = self.normalized_state_planes(onboard_state, e2.shape[-2], e2.shape[-1])
        return torch.cat((e2, state), dim=1)

    @staticmethod
    def decode_gate(packed):
        return {
            "gate_heatmap_logits": packed[:, :4],
            "gate_visibility_logits": packed[:, 4:].mean((-2, -1)),
        }

    @staticmethod
    def decode_ttc(packed):
        return {
            "inverse_ttc": F.softplus(packed[:, 0:1]),
            "inverse_depth": F.softplus(packed[:, 1:2]),
            "flow": packed[:, 2:4],
            "risk_logits": packed[:, 4:7],
        }

    def forward(self, images, onboard_state):
        if tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(f"expected image shape [B,{self.image_shape}]")
        if tuple(onboard_state.shape[1:]) != self.onboard_state_shape:
            raise ValueError(f"expected onboard shape [B,{self.onboard_state_shape}]")
        e2 = self.encoder(images)
        gate = self.gate_head(e2)
        ttc = self.ttc_head(self.pack_ttc_input(e2, onboard_state))
        return {**self.decode_gate(gate), **self.decode_ttc(ttc)}

    def initialize_from_dory_bridge(self, bridge: str | Path):
        """Warm-start the proven two-frame encoder and four corner channels."""
        bridge = Path(bridge)
        encoder_archive = np.load(bridge / "encoder_float_state.npz")
        encoder_state = self.encoder.state_dict()
        loaded_encoder = []
        for key in encoder_archive.files:
            if key in encoder_state and tuple(encoder_state[key].shape) == encoder_archive[key].shape:
                encoder_state[key] = torch.from_numpy(encoder_archive[key])
                loaded_encoder.append(key)
        self.encoder.load_state_dict(encoder_state)

        corner_archive = np.load(bridge / "corner_head_float_state.npz")
        gate_state = self.gate_head.state_dict()
        loaded_gate = []
        for key in corner_archive.files:
            destination = key.replace("head.0.", "head_features.")
            if key.startswith("head.1."):
                destination = "output." + key[len("head.1."):]
            if destination not in gate_state:
                continue
            source = torch.from_numpy(corner_archive[key])
            target = gate_state[destination]
            if destination.startswith("output."):
                target[:4].copy_(source)
                loaded_gate.append(destination + "[:4]")
            elif tuple(target.shape) == tuple(source.shape):
                gate_state[destination] = source
                loaded_gate.append(destination)
        self.gate_head.load_state_dict(gate_state)
        return {
            "bridge": str(bridge.resolve()),
            "encoder_tensors_loaded": len(loaded_encoder),
            "gate_tensors_loaded": len(loaded_gate),
            "fresh_modules": ["gate_head.output[4:8]", "ttc_head"],
        }

    def initialize_from_shallower_checkpoint(self, checkpoint: str | Path):
        """Expand a trained TTC graph with identity-initialized residual blocks."""
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = saved.get("model", saved)
        source_blocks = sorted({
            int(key.split(".")[2])
            for key in source
            if key.startswith("ttc_head.deep.") and ".block." in key
        })
        if not source_blocks:
            raise RuntimeError("checkpoint does not contain residual TTC blocks")
        source_refinements = max(source_blocks) + 1
        if source_refinements >= self.ttc_refinements:
            raise ValueError(
                f"source has {source_refinements} refinements, target has {self.ttc_refinements}"
            )
        target = self.state_dict()
        final_source_prefix = f"ttc_head.deep.{source_refinements}."
        final_target_prefix = f"ttc_head.deep.{self.ttc_refinements}."
        copied = []
        for key, value in source.items():
            destination = (
                final_target_prefix + key[len(final_source_prefix):]
                if key.startswith(final_source_prefix)
                else key
            )
            if destination in target and target[destination].shape == value.shape:
                target[destination] = value
                copied.append(destination)
        self.load_state_dict(target)
        with torch.no_grad():
            for index in range(source_refinements, self.ttc_refinements):
                block = self.ttc_head.deep[index].block
                block.pointwise[1].weight.zero_()
                block.pointwise[1].bias.zero_()
        return {
            "checkpoint": str(checkpoint.resolve()),
            "source_epoch": saved.get("epoch"),
            "source_ttc_refinements": source_refinements,
            "target_ttc_refinements": self.ttc_refinements,
            "copied_tensors": len(copied),
            "new_residual_blocks_initialized_as_identity": (
                self.ttc_refinements - source_refinements
            ),
        }


def dory_graphs(model: DoryPartitionedMotionGateTTCNet):
    return {
        "encoder": model.encoder,
        "gate_head": model.gate_head,
        "ttc_head": model.ttc_head,
    }


def load_dory_checkpoint(checkpoint: str | Path, device="cpu"):
    """Load a partitioned checkpoint and infer its TTC refinement depth."""
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    state = saved.get("model", saved)
    residual_indices = sorted({
        int(key.split(".")[2])
        for key in state
        if key.startswith("ttc_head.deep.") and ".block." in key
    })
    if not residual_indices:
        raise RuntimeError("checkpoint does not contain DORY TTC residual blocks")
    refinements = max(residual_indices) + 1
    model = DoryPartitionedMotionGateTTCNet(ttc_refinements=refinements).to(device)
    model.load_state_dict(state)
    return model, {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": saved.get("epoch"),
        "ttc_refinements": refinements,
    }


def compact_identity_ttc_blocks(model: DoryPartitionedMotionGateTTCNet):
    """Remove trailing residual blocks whose terminal BN is identically zero."""
    identity = []
    for index in range(model.ttc_refinements):
        bn = model.ttc_head.deep[index].block.pointwise[1]
        if torch.count_nonzero(bn.weight) == 0 and torch.count_nonzero(bn.bias) == 0:
            identity.append(index)
    if not identity:
        return copy.deepcopy(model), {
            "source_ttc_refinements": model.ttc_refinements,
            "deployed_ttc_refinements": model.ttc_refinements,
            "pruned_identity_blocks": [],
        }
    first = identity[0]
    expected = list(range(first, model.ttc_refinements))
    if identity != expected:
        raise RuntimeError(f"only trailing identity blocks can be compacted: {identity}")
    compact = DoryPartitionedMotionGateTTCNet(ttc_refinements=first)
    source, target = model.state_dict(), compact.state_dict()
    final_source = f"ttc_head.deep.{model.ttc_refinements}."
    final_target = f"ttc_head.deep.{first}."
    for key, value in source.items():
        destination = (
            final_target + key[len(final_source):]
            if key.startswith(final_source)
            else key
        )
        if destination in target and target[destination].shape == value.shape:
            target[destination] = value.detach().cpu().clone()
    compact.load_state_dict(target)
    compact.train(model.training)
    return compact, {
        "source_ttc_refinements": model.ttc_refinements,
        "deployed_ttc_refinements": first,
        "pruned_identity_blocks": identity,
        "function_change": "none; each pruned residual branch was identically zero",
    }
