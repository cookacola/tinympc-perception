#!/usr/bin/env python3
"""Report verified completed frames for a sharded course dataset."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()

frames = 0
shards = 0
bytes_total = 0
for shard in sorted(args.root.glob("shard_*")):
    if not (shard / "_SUCCESS").exists():
        continue
    report = json.loads((shard / "validation.json").read_text())
    if not report.get("passed"):
        continue
    frames += int(report["validated"])
    shards += 1
    bytes_total += sum(path.stat().st_size for path in shard.rglob("*") if path.is_file())

print(
    json.dumps(
        {
            "verified_shards": shards,
            "verified_frames": frames,
            "bytes": bytes_total,
            "gib": bytes_total / 2**30,
        },
        indent=2,
    )
)

