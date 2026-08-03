#!/usr/bin/env python3
"""Find the first failing DORY layer from a checksum-harness UART log.

The generated checksum harness emits one ordered checksum per fused GAP8
layer. This utility deliberately treats a partial log as inconclusive instead
of calling a final-only checksum a full layer-bisect pass.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def expected_checksums(header: str) -> list[int]:
    match = re.search(
        r"static int activations_out_checksum\[\d+\]\[1\] = \{(.*?)\};",
        header,
        flags=re.S,
    )
    if not match:
        raise ValueError("activations_out_checksum table not found")
    values = [int(item) for item in re.findall(r"\b\d+\b", match.group(1))]
    if not values:
        raise ValueError("activation checksum table is empty")
    return values


def parse_log(log: str) -> list[dict[str, object]]:
    entries = []
    pattern = re.compile(
        r"Checking\s+(?P<name>[^:]+):\s+Checksum\s+"
        r"(?P<status>OK|Failed:.*)$"
    )
    for line in log.splitlines():
        match = pattern.search(line)
        if match:
            entries.append({
                "name": match.group("name"),
                "passed": match.group("status") == "OK",
                "status": match.group("status"),
            })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expected = expected_checksums(
        (args.app_dir / "inc/gap8_network.h").read_text()
    )
    observed = parse_log(args.log.read_text(errors="replace"))
    report: dict[str, object] = {
        "expected_layers": len(expected),
        "observed_checksums": len(observed),
        "complete": len(observed) == len(expected),
        "passed": False,
    }
    if len(observed) < len(expected):
        report["status"] = "inconclusive_partial_log"
    else:
        first_failure = next(
            (index for index, item in enumerate(observed[:len(expected)])
             if not item["passed"]),
            None,
        )
        report["first_failure_layer"] = first_failure
        report["passed"] = first_failure is None
        report["status"] = "passed" if first_failure is None else "layer_mismatch"
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if report["status"] == "layer_mismatch":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
