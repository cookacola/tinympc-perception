#!/usr/bin/env python3
"""Audit gate metrics and split-specific TTC retention for a joint checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train_ttc_gate_head import finite_json
from .train_ttc_gate_joint_finetune import TEST_LIMITS, VALIDATION_LIMITS, evaluate
from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_model import MotionConditionedESPNetGateTTCNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("joint checkpoint evaluation requires a Slurm GPU allocation")
    model = MotionConditionedESPNetGateTTCNet().to(device)
    initialization = model.initialize_from_checkpoint(args.checkpoint)
    options = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization": initialization,
        "validation": evaluate(
            model, DataLoader(TTCGateDataset(args.dataset, "validation"), **options),
            device, VALIDATION_LIMITS,
        ),
        "test": evaluate(
            model, DataLoader(TTCGateDataset(args.dataset, "test"), **options),
            device, TEST_LIMITS,
        ),
    }
    args.output.write_text(json.dumps(finite_json(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
