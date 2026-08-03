#!/usr/bin/env python3
"""Export and structurally audit the sequential student's single ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from .model_sequential import SequentialSTDCNet
from .output_contract import write_c_header
from .quantization import fold_batch_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") not in (None, "SequentialSTDCNet"):
        raise RuntimeError(f"not a sequential checkpoint: {state.get('architecture')}")
    model = SequentialSTDCNet().eval()
    model.load_state_dict(state["model"])
    model = fold_batch_norms(model).eval()
    onnx_path = args.output_dir / "gap8_sequential_student.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 1, 120, 160), onnx_path,
        input_names=["hm01b0"], output_names=["perception_scores_12x15x20"],
        opset_version=13, dynamo=False,
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    allowed = {"Conv", "Relu", "Identity", "Constant"}
    unsupported = sorted(set(operators) - allowed)
    if len(graph.graph.output) != 1:
        raise RuntimeError("sequential deployment graph must have one output")
    write_c_header(args.output_dir / "perception_output_contract.h")
    report = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(onnx_path),
        "input": {"hm01b0": [1, 1, 120, 160], "type": "uint8 at deployment"},
        "output": {"perception_scores_12x15x20": [1, 12, 15, 20]},
        "operators": operators,
        "unsupported_dory_initial_ops": unsupported,
        "batch_norm_folded": "BatchNormalization" not in operators,
        "single_output_graph": True,
        "quantization_status": "FP32 structural export; NeMO integer conversion remains required",
    }
    (args.output_dir / "onnx_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if unsupported:
        raise SystemExit(f"unsupported operators: {unsupported}")


if __name__ == "__main__":
    main()
