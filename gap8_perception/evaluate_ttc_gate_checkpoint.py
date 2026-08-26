#!/usr/bin/env python3
"""Evaluate a trained TTC/gate checkpoint on natural validation and test splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train_ttc_gate_head import evaluate, finite_json
from .ttc_gate_data import TTCGateDataset
from .ttc_motion_gate_model import MotionConditionedESPNetGateTTCNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("checkpoint evaluation requires a Slurm GPU allocation")
    model = MotionConditionedESPNetGateTTCNet().to(device)
    initialization = model.initialize_from_checkpoint(args.checkpoint)
    options = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization": initialization,
        "coordinate_decoder": "20x20_heatmap_peak_cell_center",
        "validation": evaluate(
            model, DataLoader(TTCGateDataset(args.dataset, "validation"), **options), device
        ),
        "test": evaluate(
            model, DataLoader(TTCGateDataset(args.dataset, "test"), **options), device
        ),
    }
    args.output.write_text(json.dumps(finite_json(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
