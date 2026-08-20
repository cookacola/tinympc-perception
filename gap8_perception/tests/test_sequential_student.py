import json
from pathlib import Path

import torch
from torch import nn

from gap8_perception.model_sequential import SequentialSTDCNet
from gap8_perception.output_contract import decode_scalar_fields
from gap8_perception.profile_sequential import profile


def test_replacement_has_exact_single_output_contract():
    model = SequentialSTDCNet().eval()
    output = model(torch.zeros(2, 1, 120, 160))
    assert output.shape == (2, 12, 15, 20)
    offsets, confidence = decode_scalar_fields(output)
    assert offsets.shape == confidence.shape == (2, 4)


def test_replacement_resource_report_is_checked_in_envelope():
    report = profile()
    assert 70_000 <= report["parameters"] <= 180_000
    assert report["macs"] < 30_000_000
    assert report["output_nchw"] == [1, 12, 15, 20]
    assert report["macs"] == 28_152_000


def test_checked_in_architecture_manifest_matches_every_profiled_layer():
    manifest_path = Path(__file__).parents[1] / "configs" / "sequential_architecture.json"
    manifest = json.loads(manifest_path.read_text())
    report = profile()
    assert manifest["input_nchw"] == report["input_nchw"]
    assert manifest["output_nchw"] == report["output_nchw"]
    assert manifest["parameters"] == report["parameters"]
    assert manifest["macs"] == report["macs"]
    assert manifest["layers"] == [
        {
            "name": layer["name"],
            "macs": layer["macs"],
            "output_nchw": [1, *layer["output"]],
        }
        for layer in report["layers"]
    ]


def test_replacement_contains_no_banned_graph_plumbing_or_biases():
    model = SequentialSTDCNet()
    assert not any(isinstance(module, (nn.Upsample, nn.ConvTranspose2d)) for module in model.modules())
    assert all(module.bias is None for module in model.modules() if isinstance(module, nn.Conv2d))
    assert model.output.bias is None
    assert sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()) == 23
