"""Frozen two-frame ESPNetV2-Lite backbone with attachable gate heads."""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvBNReLU, DSConv


class EESPBlock(nn.Module):
    def __init__(self, cin, cout, stride, branches=4):
        super().__init__()
        hidden = cout // branches
        self.reduce = ConvBNReLU(cin, hidden, 1)
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride=stride, padding=dilation,
                          dilation=dilation, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(cout, cout, 1, groups=branches, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        reduced = self.reduce(x)
        outputs, running = [], None
        for path in self.paths:
            current = path(reduced)
            running = current if running is None else running + current
            outputs.append(running)
        y = self.project(torch.cat(outputs, 1))
        return F.relu(x + y if self.residual else y, inplace=True)


class ESPNetV2LiteEncoder(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 16, 3, 2)
        self.stage1 = nn.Sequential(EESPBlock(16, 32, 2), EESPBlock(32, 32, 1))
        self.stage2 = nn.Sequential(EESPBlock(32, 64, 2), EESPBlock(64, 64, 1))
        self.stage3 = nn.Sequential(EESPBlock(64, 96, 2), EESPBlock(96, 96, 1))

    def forward(self, x):
        x = self.stem(x)
        early = self.stage1(x)
        middle = self.stage2(early)
        return early, middle, self.stage3(middle)


def encoder_fingerprint(encoder):
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenESPNetHead(nn.Module):
    """Train only a common-capacity head at an early, middle, or late tap."""
    def __init__(self, checkpoint, tap, task, representation):
        super().__init__()
        if tap not in {"early", "middle", "late"}:
            raise ValueError(tap)
        if task not in {"corner", "gate"}:
            raise ValueError(task)
        if representation not in {"heatmap", "direct", "binary", "sdf"}:
            raise ValueError(representation)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("encoder") != "espnetv2_lite" or saved.get("history") != 2:
            raise ValueError("checkpoint must be a two-frame ESPNetV2-Lite model")
        self.encoder = ESPNetV2LiteEncoder(2)
        state = {key.removeprefix("encoder."): value for key, value in saved["model"].items()
                 if key.startswith("encoder.")}
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.backbone_fingerprint = encoder_fingerprint(self.encoder)
        self.tap, self.task, self.representation = tap, task, representation
        channels = {"early": 32, "middle": 64, "late": 96}[tap]
        self.adapter = ConvBNReLU(channels, 32, 1)
        outputs = 4 if task == "corner" and representation == "heatmap" else 8 if task == "corner" else 1
        self.head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, outputs, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        # Synthetic gate poses are independent samples, so repeat the current
        # grayscale image rather than pretending adjacent files are a sequence.
        with torch.no_grad():
            features = self.encoder(image.repeat(1, 2, 1, 1))
        feature = features[{"early": 0, "middle": 1, "late": 2}[self.tap]]
        output = self.head(self.adapter(feature))
        if output.shape[-1] != 40:
            output = F.interpolate(output, size=(40, 40), mode="bilinear", align_corners=False)
        if self.task == "corner" and self.representation == "direct":
            output = F.adaptive_avg_pool2d(output, 1).flatten(1).reshape(-1, 4, 2)
        return output

    def assert_backbone_unchanged(self):
        current = encoder_fingerprint(self.encoder)
        if current != self.backbone_fingerprint:
            raise RuntimeError("frozen ESPNet backbone state changed")
        return current
