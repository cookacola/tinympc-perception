#!/usr/bin/env python3
"""Create a deterministic shard/seed-level 40k/5k/5k split."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    shards = [
        path.name
        for path in sorted(args.dataset.glob("shard_*"))
        if (path / "_SUCCESS").is_file()
    ]
    if len(shards) != 50:
        raise RuntimeError(f"expected 50 successful shards, found {len(shards)}")
    random.Random(args.seed).shuffle(shards)
    split = {
        "seed": args.seed,
        "grouping": "capture shard (one deterministic generator seed per shard)",
        "warning": "all groups share one fixed scene; this prevents shard/seed leakage but cannot test scene generalization",
        "train": sorted(shards[:40]),
        "validation": sorted(shards[40:45]),
        "test": sorted(shards[45:]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in split.items() if isinstance(value, list)}))


if __name__ == "__main__":
    main()
