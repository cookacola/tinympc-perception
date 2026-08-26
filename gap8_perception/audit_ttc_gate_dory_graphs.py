#!/usr/bin/env python3
"""Export and statically audit the three stock-DORY gate/TTC graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from .ttc_motion_gate_dory_model import DoryPartitionedMotionGateTTCNet, dory_graphs


GRAPH_INPUTS = {
    "encoder": (1, 2, 160, 160),
    "gate_head": (1, 64, 20, 20),
    "ttc_head": (1, 74, 20, 20),
}
GRAPH_OUTPUTS = {
    "encoder": [1, 64, 20, 20],
    "gate_head": [1, 8, 20, 20],
    "ttc_head": [1, 7, 20, 20],
}
ALLOWED_OPERATORS = {"Conv", "Relu", "Add", "Identity", "Constant"}


def audit_graphs(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, graph in dory_graphs(model).items():
        graph.eval()
        path = output_dir / f"{name}_float.onnx"
        torch.onnx.export(
            graph, torch.zeros(GRAPH_INPUTS[name]), path,
            input_names=[f"{name}_input"], output_names=[f"{name}_output"],
            opset_version=13, dynamo=False,
        )
        exported = onnx.load(path)
        onnx.checker.check_model(exported)
        operators = sorted({node.op_type for node in exported.graph.node})
        unsupported = sorted(set(operators) - ALLOWED_OPERATORS)
        output_shape = [
            dimension.dim_value
            for dimension in exported.graph.output[0].type.tensor_type.shape.dim
        ]
        reports[name] = {
            "onnx": str(path.resolve()),
            "inputs": len(exported.graph.input),
            "outputs": len(exported.graph.output),
            "input_shape": list(GRAPH_INPUTS[name]),
            "output_shape": output_shape,
            "operators": operators,
            "unsupported_operators": unsupported,
            "passed": (
                len(exported.graph.input) == 1
                and len(exported.graph.output) == 1
                and output_shape == GRAPH_OUTPUTS[name]
                and not unsupported
            ),
        }
    return {
        "format": "dory-partitioned-motion-gate-ttc-static-audit-v1",
        "allowed_operators": sorted(ALLOWED_OPERATORS),
        "graphs": reports,
        "passed": all(report["passed"] for report in reports.values()),
        "remaining_acceptance": [
            "NeMO int8 quantization and integer parity",
            "installed DORY frontend and GAP8 tiler",
            "GVSOC checksum parity",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = DoryPartitionedMotionGateTTCNet()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("model", saved)
        blocks = {
            int(key.split(".")[2])
            for key in state
            if key.startswith("ttc_head.deep.") and ".block." in key
        }
        model = DoryPartitionedMotionGateTTCNet(
            ttc_refinements=max(blocks) + 1 if blocks else 3
        )
        model.load_state_dict(state)
    report = audit_graphs(model.eval(), args.output_dir)
    (args.output_dir / "static_dory_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("one or more graphs failed the stock-DORY structural contract")


if __name__ == "__main__":
    main()
