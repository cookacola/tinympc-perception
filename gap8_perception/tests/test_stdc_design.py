import torch

from gap8_perception.losses_stdc import design_multitask_loss
from gap8_perception.model_stdc import Gap8STDCMultiHeadNet, ProposedSTDCFPNNet
from gap8_perception.profile_stdc import profile


def test_design_output_shapes_and_corner_confidence():
    model = Gap8STDCMultiHeadNet().eval()
    image = torch.zeros(2, 1, 120, 160)
    outputs = model(image)
    predictions = model.predict(image)
    assert outputs["corners"].shape == (2, 4, 30, 40)
    assert outputs["danger"].shape == (2, 1, 15, 20)
    assert model.forward_packed(image).shape == (2, 5, 30, 40)
    assert predictions["corner_confidence"].shape == (2, 4)
    assert ((predictions["danger"] >= 0) & (predictions["danger"] <= 1)).all()


def test_design_meets_parameter_and_mac_budgets():
    report = profile(Gap8STDCMultiHeadNet().eval())
    assert 100_000 <= report["parameters"] <= 180_000
    assert report["macs"] <= 30_000_000


def test_design_loss_backpropagates_both_heads():
    model = Gap8STDCMultiHeadNet()
    outputs = model(torch.rand(2, 1, 120, 160))
    batch = {
        "corners": torch.zeros(2, 4, 30, 40),
        "corner_valid": torch.tensor([False, False]),
        "danger": torch.zeros(2, 1, 15, 20),
    }
    losses = design_multitask_loss(outputs, batch)
    losses["total"].backward()
    assert model.corner_logits.weight.grad is not None
    assert model.danger_logits.weight.grad is not None
    assert model.stem[0][0].weight.grad is not None


def test_only_expected_learned_spatial_operators():
    model = Gap8STDCMultiHeadNet()
    forbidden = (torch.nn.ConvTranspose2d,)
    assert not any(isinstance(module, forbidden) for module in model.modules())
def test_proposed_fpn_shapes_and_training_only_heads():
    model = ProposedSTDCFPNNet()
    model.train()
    output = model(torch.zeros(2, 1, 120, 160))
    assert output["corners"].shape == (2, 4, 30, 40)
    assert output["danger"].shape == (2, 1, 30, 40)
    assert output["boundary"].shape == (2, 1, 30, 40)
    assert output["global_coordinates"].shape == (2, 8)
    model.eval()
    assert set(model(torch.zeros(1, 1, 120, 160))) == {"corners", "danger"}
