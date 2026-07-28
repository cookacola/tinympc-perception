#!/usr/bin/env python3
"""Export a BN-folded ONNX graph and validate GAP8 operator constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from gap8_perception.model import Gap8MultiTaskNet
from gap8_perception.quantization import fold_batch_norms, prepare_int8_qat


class ExportWrapper(nn.Module):
    def __init__(self, model, include_gate=True):
        super().__init__()
        self.model = model
        self.include_gate = include_gate

    def forward(self, image):
        return self.model.forward_packed(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--drop-gate-head", action="store_true",
        help="Export the auxiliary-head ablation with only corners and danger.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gate_head = state.get("gate_head", True) and not args.drop_gate_head
    state_dim = state.get("state_dim", 8)
    base = Gap8MultiTaskNet(gate_head, state_dim)
    qparams = {}
    if state.get("quantization_aware"):
        qat = prepare_int8_qat(base)
        if args.drop_gate_head:
            target_state = qat.state_dict()
            compatible = {
                key: value
                for key, value in state["model"].items()
                if key in target_state
                and target_state[key].shape == value.shape
            }
            qat.load_state_dict(compatible, strict=False)
        else:
            qat.load_state_dict(state["model"])
        qat.eval()
        # Apply learned weight fake quantization, then transfer core tensors.
        with torch.no_grad():
            for name, module in qat.named_modules():
                if isinstance(module, (nn.qat.Conv2d, nn.qat.Linear)):
                    module.weight.copy_(module.weight_fake_quant(module.weight))
                if hasattr(module, "activation_post_process") and hasattr(
                    module.activation_post_process, "scale"
                ):
                    qparams[name] = {
                        "scale": module.activation_post_process.scale.detach().cpu().tolist(),
                        "zero_point": module.activation_post_process.zero_point.detach().cpu().tolist(),
                    }
        core = base.state_dict()
        core.update({key: value for key, value in qat.state_dict().items() if key in core})
        base.load_state_dict(core)
    else:
        if args.drop_gate_head:
            target_state = base.state_dict()
            compatible = {
                key: value
                for key, value in state["model"].items()
                if key in target_state
                and target_state[key].shape == value.shape
            }
            base.load_state_dict(compatible, strict=False)
        else:
            base.load_state_dict(state["model"])
    model = fold_batch_norms(base)
    path = args.output_dir / "gap8_multitask.onnx"
    torch.onnx.export(
        ExportWrapper(model, include_gate=gate_head).eval(),
        torch.zeros(1, 1, 160, 160),
        path,
        input_names=["hm01b0"],
        output_names=["packed_multitask_logits_40x40"],
        opset_version=13,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    # This is the actual operator surface accepted by the installed DORY
    # Quantlab frontend, not a generic "GAP8-friendly" approximation.
    allowed = {
        "Conv", "Relu", "Add", "Identity", "Constant",
    }
    unsupported = sorted(set(operators) - allowed)
    report = {
        "onnx": str(path),
        "operators": operators,
        "shape_only_operators": sorted(
            set(operators) & {"Constant", "Identity", "Reshape", "Unsqueeze"}
        ),
        "unsupported_gap8_initial_ops": unsupported,
        "batch_norm_folded": "BatchNormalization" not in operators,
        "inputs": {
            "hm01b0": [1, 1, 160, 160],
        },
        "outputs": {
            "packed_multitask_logits_40x40": [
                1, 8 if gate_head else 7, 40, 40
            ],
        },
        "logical_channel_layout": (
            ["corner_tl", "corner_tr", "corner_br", "corner_bl",
             "nominal_collision", "inverse_range", "uncertainty", "gate_opening"]
            if gate_head
            else ["corner_tl", "corner_tr", "corner_br", "corner_bl",
                  "nominal_collision", "inverse_range", "uncertainty"]
        ),
        "fc_postprocess": (
            "split channels; 2x2 average control logits to 20x20; sigmoid; "
            "compute speed-conditioned danger and TTC"
        ),
        "gate_head_retained": gate_head,
        "deployment_frontend": "DORY Quantlab",
        "dory_graph_inputs": 1,
        "motion_conditioning": (
            "controller shifts nominal-speed collision logits and combines "
            "them with range, velocity, horizon, and latency"
        ),
        "measured_gap8_latency": None,
        "note": "DORY is installed locally; final code generation requires a DORY-pattern quantized ONNX plus GAP SDK/GAP8 validation.",
    }
    (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "qat_qparams.json").write_text(json.dumps(qparams, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if unsupported:
        raise SystemExit(f"unsupported operators: {unsupported}")


if __name__ == "__main__":
    main()
