#!/usr/bin/env python3
"""Fail when canonical sequential entrypoints drift from the completed design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model_sequential import SequentialSTDCNet
from .output_contract import NORMAL_ANGLES_DEG, OUTPUT_CHANNELS
from .profile_sequential import profile


CANONICAL_MODULES = (
    "model_sequential.py",
    "data_sequential.py",
    "losses_sequential.py",
    "train_sequential.py",
    "train_sequential_qat.py",
    "evaluate_sequential.py",
    "controller_sequential.py",
    "generate_sequential_targets.py",
    "export_sequential_onnx.py",
    "nemo_sequential_integer_export.py",
)
FORBIDDEN_CONTRACT_TERMS = (
    "speed_variants_mps",
    "vehicle_state_f32",
    "danger_u8",
    "inverse_range_u8",
    "MultiTaskDataset",
)


def audit(repo: Path) -> dict:
    package = repo / "gap8_perception"
    model = SequentialSTDCNet().eval()
    output = model(torch.zeros(1, 1, 120, 160))
    if tuple(output.shape) != (1, 12, 15, 20):
        raise RuntimeError(f"wrong output shape: {tuple(output.shape)}")
    if OUTPUT_CHANNELS != 12 or len(NORMAL_ANGLES_DEG) != 4:
        raise RuntimeError("output constants violate the design")
    banned_modules = (nn.Upsample, nn.ConvTranspose2d)
    if any(isinstance(module, banned_modules) for module in model.modules()):
        raise RuntimeError("canonical model contains a banned operator")
    for module_name in CANONICAL_MODULES:
        text = (package / module_name).read_text()
        found = [term for term in FORBIDDEN_CONTRACT_TERMS if term in text]
        if found:
            raise RuntimeError(f"{module_name} contains legacy contract terms: {found}")
    for script in package.glob("run_sequential*.slurm"):
        text = script.read_text()
        if "generate_targets.py" in text or "gap8_sequential_targets_75k_v2" in text:
            raise RuntimeError(f"{script.name} references a stale target path")
    design = (repo / "docs/Completed_CNN_Design_Document.md").read_text()
    for phrase in (
        "Y_t \\in \\mathbb{Z}^{1 \\times 12 \\times 15 \\times 20}",
        "Four fixed-normal half-space offset fields",
        "gate-frame centerline",
    ):
        if phrase not in design:
            raise RuntimeError(f"design document is missing {phrase!r}")
    resources = profile()
    if resources["macs"] >= 30_000_000 or resources["parameters"] > 180_000:
        raise RuntimeError("canonical model exceeds the design resource envelope")
    return {
        "passed": True,
        "input_nchw": [1, 1, 120, 160],
        "output_nchw": [1, 12, 15, 20],
        "parameters": resources["parameters"],
        "macs": resources["macs"],
        "fixed_normal_angles_deg": list(NORMAL_ANGLES_DEG),
        "canonical_modules_checked": list(CANONICAL_MODULES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.repo)
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
