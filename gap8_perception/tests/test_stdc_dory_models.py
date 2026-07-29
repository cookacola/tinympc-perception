import torch

from gap8_perception.model_stdc_dory import (
    Gap8STDCCornerDoryNet,
    Gap8STDCDangerDoryNet,
    Gap8STDCSharedDoryNet,
    initialize_shared_from_pair,
    shared_deployment_graphs,
)
from gap8_perception.profile_stdc_dory import combined_profile, shared_profile


def test_resize_free_dory_shapes_and_combined_budget():
    image = torch.zeros(1, 1, 120, 160)
    assert Gap8STDCCornerDoryNet()(image).shape == (1, 4, 30, 40)
    assert Gap8STDCDangerDoryNet()(image).shape == (1, 1, 8, 10)
    report = combined_profile()
    assert 100_000 <= report["combined_parameters"] <= 180_000
    assert report["combined_macs"] <= 30_000_000


def test_dory_students_have_no_resize_or_concat():
    for model in (Gap8STDCCornerDoryNet(), Gap8STDCDangerDoryNet()):
        names = set(model.forward.__code__.co_names)
        assert "interpolate" not in names
        assert "cat" not in names


def test_shared_dory_graph_split_is_numerically_exact():
    model = Gap8STDCSharedDoryNet().eval()
    encoder, corner, danger = shared_deployment_graphs(model)
    image = torch.randn(2, 1, 120, 160)
    with torch.no_grad():
        expected = model(image)
        shared = encoder(image)
        actual_corner = corner(shared)
        actual_danger = danger(shared)
    assert shared.shape == (2, 32, 30, 40)
    torch.testing.assert_close(actual_corner, expected["corners"])
    torch.testing.assert_close(actual_danger, expected["danger"])


def test_shared_dory_profile_is_inside_design_envelope():
    report = shared_profile()
    assert 100_000 <= report["combined_parameters"] <= 180_000
    assert report["combined_macs"] < 30_000_000
    assert report["combined_macs"] < combined_profile()["combined_macs"]


def test_shared_pair_initialization_preserves_corner_graph():
    corner = Gap8STDCCornerDoryNet().eval()
    danger = Gap8STDCDangerDoryNet().eval()
    shared = Gap8STDCSharedDoryNet().eval()
    initialize_shared_from_pair(
        shared, corner.state_dict(), danger.state_dict()
    )
    image = torch.randn(1, 1, 120, 160)
    with torch.no_grad():
        torch.testing.assert_close(
            shared(image)["corners"], corner(image)
        )
