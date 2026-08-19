import torch

from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.train_gate_tap_ablation import GateTapNet
from gap8_perception.espnet_frozen_heads import ESPNetV2LiteEncoder, encoder_fingerprint


def test_output_shapes_with_gate():
    outputs = Gap8MultiTaskNet(gate_head=True)(
        torch.zeros(2, 1, 160, 160), torch.zeros(2, 8)
    )
    assert outputs["corners"].shape == (2, 4, 40, 40)
    assert outputs["danger"].shape == (2, 1, 20, 20)
    assert outputs["urgency"].shape == (2, 1, 20, 20)
    assert outputs["uncertainty"].shape == (2, 1, 20, 20)
    assert outputs["gate"].shape == (2, 1, 40, 40)


def test_gate_head_is_removable():
    outputs = Gap8MultiTaskNet(gate_head=False)(torch.zeros(1, 1, 160, 160))
    assert set(outputs) == {"corners", "danger", "urgency", "uncertainty"}


def test_uses_only_deployment_friendly_spatial_ops():
    model = Gap8MultiTaskNet()
    forbidden = (torch.nn.ConvTranspose2d, torch.nn.Upsample)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert "interpolate" not in model.forward_image.__code__.co_names


def test_dory_inference_path_is_single_input():
    model = Gap8MultiTaskNet().eval()
    output = model.forward_packed(torch.zeros(1, 1, 160, 160))
    assert output.shape == (1, 8, 40, 40)


def test_packed_no_gate_layout_has_seven_channels():
    output = Gap8MultiTaskNet(gate_head=False).forward_packed(
        torch.zeros(1, 1, 160, 160)
    )
    assert output.shape == (1, 7, 40, 40)


def test_cnn_is_image_only_for_stock_dory_frontend():
    torch.manual_seed(7)
    model = Gap8MultiTaskNet().eval()
    image = torch.rand(1, 1, 160, 160)
    slow = torch.tensor([[0.5, 0, 0, 0, 0, 0, 1.0, 0.08]])
    fast = torch.tensor([[5.0, 0, 0, 0, 0, 0, 1.0, 0.08]])
    with torch.no_grad():
        slow_output = model(image, slow)
        fast_output = model(image, fast)
    assert torch.allclose(slow_output["danger"], fast_output["danger"])
    assert torch.allclose(slow_output["urgency"], fast_output["urgency"])
    assert torch.allclose(slow_output["corners"], fast_output["corners"])
    assert torch.allclose(slow_output["gate"], fast_output["gate"])


def test_gate_taps_produce_common_resolution_and_gradients():
    for tap in ("early80", "mid40", "late40"):
        model = GateTapNet(tap)
        output = model(torch.rand(2, 1, 160, 160))
        assert output.shape == (2, 1, 40, 40)
        output.mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_espnet_taps_and_fingerprint_are_stable():
    model = ESPNetV2LiteEncoder(2).eval()
    before = encoder_fingerprint(model)
    early, middle, late = model(torch.rand(2, 2, 160, 160))
    assert early.shape == (2, 32, 40, 40)
    assert middle.shape == (2, 64, 20, 20)
    assert late.shape == (2, 96, 10, 10)
    assert encoder_fingerprint(model) == before
