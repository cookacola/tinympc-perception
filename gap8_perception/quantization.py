"""QAT preparation and BatchNorm-folding helpers."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat


def prepare_int8_qat(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).train()
    model.qconfig = get_default_qat_qconfig("qnnpack")
    return prepare_qat(model, inplace=True)


def fold_batch_norms(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            for index in range(len(module) - 1):
                if isinstance(module[index], nn.Conv2d) and isinstance(
                    module[index + 1], nn.BatchNorm2d
                ):
                    module[index] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                        module[index], module[index + 1]
                    )
                    module[index + 1] = nn.Identity()
    return model
