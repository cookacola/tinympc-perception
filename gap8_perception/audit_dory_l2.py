#!/usr/bin/env python3
"""Statically verify directional L2 allocation in generated DORY graphs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARRAY_NAMES = (
    "activations_size",
    "activations_out_size",
    "weights_size",
    "layer_with_weights",
    "L3_input_layers",
    "L3_output_layers",
    "branch_input",
    "branch_output",
    "branch_change",
)


def parse_array(text: str, name: str) -> list[int]:
    match = re.search(
        rf"static int {re.escape(name)}\[\d+\] = \{{(.*?)\}};",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing generated array {name}")
    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


def simulate_directional_allocator(arrays: dict[str, list[int]]) -> dict:
    count = len(arrays["activations_size"])
    if any(len(values) != count for values in arrays.values()):
        raise ValueError("generated allocation arrays have inconsistent lengths")
    begin = end_used = peak = bypass_dimension = 0
    direction = 1
    trace = []

    def allocate(size: int, allocation_direction: int):
        nonlocal begin, end_used, peak
        if allocation_direction:
            begin += size
        else:
            end_used += size
        peak = max(peak, begin + end_used)

    def release(size: int, allocation_direction: int):
        nonlocal begin, end_used
        if allocation_direction:
            begin -= size
        else:
            end_used -= size
        if begin < 0 or end_used < 0:
            raise ValueError("generated directional frees are not stack-balanced")

    for index in range(count):
        allocate(arrays["activations_out_size"][index], not direction)
        if arrays["L3_input_layers"][index]:
            allocate(arrays["activations_size"][index], direction)
        if arrays["layer_with_weights"][index]:
            allocate(arrays["weights_size"][index], direction)
            release(arrays["weights_size"][index], direction)
        release(arrays["activations_size"][index], direction)
        if arrays["branch_input"][index]:
            release(bypass_dimension, direction)
        if index < count - 1:
            if arrays["branch_input"][index + 1]:
                allocate(bypass_dimension, not direction)
            if arrays["branch_output"][index] or arrays["branch_change"][index]:
                bypass_dimension = arrays["activations_out_size"][index]
            if arrays["branch_change"][index]:
                release(arrays["activations_out_size"][index], not direction)
                allocate(arrays["activations_size"][index + 1], not direction)
            if arrays["L3_output_layers"][index]:
                release(arrays["activations_out_size"][index], not direction)
        trace.append(
            {
                "layer": index,
                "begin_bytes": begin,
                "end_bytes": end_used,
                "live_bytes": begin + end_used,
                "peak_bytes": peak,
            }
        )
        direction = not direction
    return {
        "layers": count,
        "peak_live_bytes": peak,
        "final_begin_bytes": begin,
        "final_end_bytes": end_used,
        "trace": trace,
    }


def audit_package(package: Path) -> dict:
    manifest = json.loads((package / "manifest.json").read_text())
    configured = manifest["memory"]["nanocockpit_workspace_configured_bytes"]
    graphs = {}
    for header in sorted((package / "inc").glob("stdc_*_network.h")):
        graph = header.stem.removeprefix("stdc_").removesuffix("_network")
        text = header.read_text()
        arrays = {name: parse_array(text, name) for name in ARRAY_NAMES}
        graphs[graph] = simulate_directional_allocator(arrays)
    if not graphs:
        raise ValueError("no namespaced DORY graph headers found")
    peak = max(item["peak_live_bytes"] for item in graphs.values())
    # dmalloc uses a strict '<' comparison, so one spare byte is required.
    required = peak + 1
    return {
        "format": "dory-directional-l2-audit-v1",
        "package": str(package.resolve()),
        "graphs": graphs,
        "peak_live_bytes": peak,
        "strict_allocator_required_bytes": required,
        "configured_workspace_bytes": configured,
        "margin_bytes": configured - required,
        "passed": configured >= required,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_package(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
