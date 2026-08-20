#!/usr/bin/env python3
"""Compose independently selected corner and danger DORY checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corner", type=Path, required=True)
    parser.add_argument("--danger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corner-selection", required=True)
    parser.add_argument("--danger-selection", required=True)
    args = parser.parse_args()
    corner = torch.load(args.corner, map_location="cpu", weights_only=False)
    danger = torch.load(args.danger, map_location="cpu", weights_only=False)
    selected = dict(corner)
    selected["corner_model"] = corner["corner_model"]
    selected["danger_model"] = danger["danger_model"]
    selected["selection"] = {
        "corner_checkpoint": str(args.corner.resolve()),
        "danger_checkpoint": str(args.danger.resolve()),
        "corner_selection": args.corner_selection,
        "danger_selection": args.danger_selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
