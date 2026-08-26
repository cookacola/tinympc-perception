#!/usr/bin/env python3
"""Write an exactly equivalent checkpoint with trailing identity TTC blocks removed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .train_ttc_gate_head import atomic_torch_save
from .ttc_motion_gate_dory_model import compact_identity_ttc_blocks, load_dory_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_model, initialization = load_dory_checkpoint(args.checkpoint)
    source_model.eval()
    compact, compaction = compact_identity_ttc_blocks(source_model)
    compact.eval()
    generator = torch.Generator().manual_seed(20260905)
    images = torch.randn(4, 2, 160, 160, generator=generator)
    onboard = torch.randn(4, 10, generator=generator)
    with torch.no_grad():
        source_output = source_model(images, onboard)
        compact_output = compact(images, onboard)
    maximum_difference = max(
        float((source_output[name] - compact_output[name]).abs().max())
        for name in source_output
    )
    if maximum_difference != 0.0:
        raise RuntimeError(f"compaction changed outputs: max difference {maximum_difference}")
    record = {
        "epoch": source_saved.get("epoch"),
        "model": compact.state_dict(),
        "config": source_saved.get("config"),
        "deployment": {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "source_initialization": initialization,
            "compaction": compaction,
            "random_parity_seed": 20260905,
            "random_parity_examples": 4,
            "maximum_absolute_output_difference": maximum_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(record, args.output)
    report = {
        **record["deployment"],
        "output_checkpoint": str(args.output.resolve()),
        "output_checkpoint_sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
