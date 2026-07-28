#!/usr/bin/env python3
"""Combined resource report for the two resize-free DORY graphs."""

from __future__ import annotations

import json

from .model_stdc_dory import Gap8STDCCornerDoryNet, Gap8STDCDangerDoryNet
from .profile_stdc import profile


def combined_profile():
    corner = profile(Gap8STDCCornerDoryNet().eval())
    danger = profile(Gap8STDCDangerDoryNet().eval())
    return {
        "corner_graph": corner,
        "danger_graph": danger,
        "combined_parameters": corner["parameters"] + danger["parameters"],
        "combined_macs": corner["macs"] + danger["macs"],
        "learned_operators": ["Conv", "ReLU", "Add"],
        "contains_resize": False,
        "contains_concat": False,
    }


if __name__ == "__main__":
    print(json.dumps(combined_profile(), indent=2))
