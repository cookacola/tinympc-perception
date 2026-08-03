#!/usr/bin/env python3
"""Create a deterministic shard/seed-level train/validation/test split."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-shards", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.expected_shards <= 2:
        raise ValueError("--expected-shards must be greater than 2")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    if args.train_fraction + args.validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")
    shards = [
        path.name
        for path in sorted(args.dataset.glob("shard_*"))
        if (path / "_SUCCESS").is_file()
    ]
    if len(shards) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} successful shards, found {len(shards)}"
        )
    random.Random(args.seed).shuffle(shards)
    train_count = round(len(shards) * args.train_fraction)
    validation_count = round(len(shards) * args.validation_fraction)
    test_start = train_count + validation_count
    if min(train_count, validation_count, len(shards) - test_start) <= 0:
        raise RuntimeError("split fractions produced an empty partition")
    split = {
        "seed": args.seed,
        "grouping": "capture shard (one deterministic generator seed per shard)",
        "train": sorted(shards[:train_count]),
        "validation": sorted(shards[train_count:test_start]),
        "test": sorted(shards[test_start:]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in split.items() if isinstance(value, list)}))


if __name__ == "__main__":
    main()
