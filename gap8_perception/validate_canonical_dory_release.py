#!/usr/bin/env python3
"""Reject DORY applications that need non-canonical quantization lowering.

This gate deliberately validates the generated application as well as the
integer ONNX fixture.  An ONNX graph can be algebraically valid while its
lowering asks a stock GAP8 PULP-NN kernel to perform an operation it does not
implement (most importantly a post-residual-add multiplier).
"""

from __future__ import print_function

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _vector(header, name):
    match = re.search(
        r"static int %s\[\d+\] = \{([^}]*)\};" % re.escape(name),
        header,
        flags=re.S,
    )
    if not match:
        raise RuntimeError("missing %s in generated network header" % name)
    return [int(value.strip()) for value in match.group(1).split(",") if value.strip()]


def _int8_convolutions(onnx_path):
    graph = onnx.load(str(onnx_path))
    initializers = {
        item.name: numpy_helper.to_array(item) for item in graph.graph.initializer
    }
    report = []
    for node in graph.graph.node:
        if node.op_type != "Conv":
            continue
        weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if weight is None:
            raise RuntimeError("missing Conv initializer for %s" % node.name)
        if not np.array_equal(weight, np.rint(weight)):
            raise RuntimeError("non-integer Conv weights in %s" % node.input[1])
        minimum, maximum = int(weight.min()), int(weight.max())
        if minimum < -128 or maximum > 127:
            raise RuntimeError("int8 overflow in %s: [%d, %d]" % (node.input[1], minimum, maximum))
        report.append({"name": node.input[1], "minimum": minimum, "maximum": maximum})
    if not report:
        raise RuntimeError("integer ONNX contains no Conv weights")
    return report


def validate(app_dir, onnx_path):
    app_dir = Path(app_dir)
    header_path = app_dir / "inc/gap8_network.h"
    header = header_path.read_text()
    names = re.search(r"static char \* Layers_name\[\d+\] = \{([^;]*)\};", header, re.S)
    if not names:
        raise RuntimeError("missing generated layer names")
    layer_names = re.findall(r'"([^"]+)"', names.group(1))
    out_mult = _vector(header, "out_mult_vector")
    out_shift = _vector(header, "out_shift_vector")
    if not (len(layer_names) == len(out_mult) == len(out_shift)):
        raise RuntimeError("generated layer metadata lengths differ")

    violations = []
    for index, name in enumerate(layer_names):
        # Stock PULP-NN's 8-bit residual-add kernel implements only its two
        # input multipliers and one pre-shift.  A non-unit output multiplier
        # would silently be ignored by generated C.
        if "QAddition" in name and out_mult[index] != 1:
            violations.append({
                "layer": index,
                "name": name,
                "reason": "post-add multiplier is not supported by stock 8-bit PULP-NN",
                "out_mult": out_mult[index],
                "out_shift": out_shift[index],
            })
        if "BNRelu" in name and out_mult[index] != 1:
            violations.append({
                "layer": index,
                "name": name,
                "reason": "BN requantization must be folded into k/lambda/shift",
                "out_mult": out_mult[index],
                "out_shift": out_shift[index],
            })

    expected = len(layer_names)
    golden_count = 0
    while (onnx_path.parent / ("out_layer%d.txt" % golden_count)).is_file():
        golden_count += 1
    if golden_count != expected:
        violations.append({
            "reason": "golden activation count does not match generated layers",
            "golden_count": golden_count,
            "generated_layers": expected,
        })
    if not (onnx_path.parent / "input.txt").is_file():
        violations.append({"reason": "missing integer input fixture"})

    return {
        "passed": not violations,
        "app_dir": str(app_dir),
        "onnx": str(onnx_path),
        "layers": layer_names,
        "out_mult_vector": out_mult,
        "out_shift_vector": out_shift,
        "int8_convolutions": _int8_convolutions(onnx_path),
        "golden_activation_count": golden_count,
        "violations": violations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.app_dir, args.onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("canonical DORY release gate failed")


if __name__ == "__main__":
    main()
