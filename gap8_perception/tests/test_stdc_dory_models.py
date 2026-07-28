import torch

from gap8_perception.model_stdc_dory import (
    Gap8STDCCornerDoryNet,
    Gap8STDCDangerDoryNet,
)
from gap8_perception.profile_stdc_dory import combined_profile


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
