#!/usr/bin/env python
"""Python-3.7 NeMO export for a shared encoder and two DORY heads."""

from __future__ import print_function

import argparse
import copy
import json
from pathlib import Path

import cv2
import nemo
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from gap8_perception.nemo_stdc_dory_export import (
    ConvBNReLU,
    DSConv,
    Stage,
    image_tensor,
)


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(1, 16, 3, 2), DSConv(16, 16)
        )
        self.stage1 = Stage(16, 32, 2)

    def forward(self, image):
        return self.stage1(self.stem(image))


class CornerHeadNet(nn.Module):
    output_channels = 4
    input_shape = (32, 30, 40)

    def __init__(self):
        super(CornerHeadNet, self).__init__()
        self.head_features = DSConv(32, 16)
        self.output_proj = ConvBNReLU(16, 4, 1)

    def forward_features(self, shared):
        return self.head_features(shared)

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


class DangerHeadNet(nn.Module):
    output_channels = 1
    input_shape = (32, 30, 40)

    def __init__(self):
        super(DangerHeadNet, self).__init__()
        self.stage2 = Stage(32, 64, 3)
        self.stage3 = Stage(64, 96, 7)
        self.output_proj = ConvBNReLU(96, 1, 1)

    def forward_features(self, shared):
        return self.stage3(self.stage2(shared))

    def forward_logits(self, shared):
        features = self.forward_features(shared)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, shared):
        return self.output_proj(self.forward_features(shared))


def load_archive(model, archive_path):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    for key in archive.files:
        if key in state:
            state[key] = torch.from_numpy(archive[key])
    model.load_state_dict(state)


def load_head(model, archive_path, graph_name):
    archive = np.load(str(archive_path))
    state = model.state_dict()
    head_prefix = "head."
    fine_head = graph_name in ("corner_head", "gate_head")
    terminal_prefix = "head.1." if fine_head else "head."
    for key in archive.files:
        if key.startswith(terminal_prefix):
            continue
        destination = (
            key.replace("head.0.", "head_features.")
            if fine_head
            else key
        )
        if destination in state:
            state[destination] = torch.from_numpy(archive[key])
    weight_key = "head.1.weight" if fine_head else "head.weight"
    bias_key = "head.1.bias" if fine_head else "head.bias"
    state["output_proj.0.weight"] = torch.from_numpy(archive[weight_key])
    eps = model.output_proj[1].eps
    state["output_proj.1.weight"] = torch.full_like(
        state["output_proj.1.weight"], (1.0 + eps) ** 0.5
    )
    state["output_proj.1.bias"] = torch.from_numpy(archive[bias_key])
    state["output_proj.1.running_mean"].zero_()
    state["output_proj.1.running_var"].fill_(1.0)
    model.load_state_dict(state)


def activation_epsilon(model):
    values = [
        float(module.eps_out)
        for module in model.modules()
        if hasattr(module, "eps_out")
    ]
    if not values:
        raise RuntimeError("integer graph exposes no output epsilon")
    return values[-1]


def bound_int8_convolution_weights(model):
    """Make the NeMO/DORY signed-weight ABI explicit and lossless.

    NeMO keeps integer-valued convolution weights in float tensors.  The ONNX
    exporter therefore used to retain values outside signed int8 while DORY
    serialized each value as one byte, silently wrapping it.  Round and clamp
    once, before both the golden fixtures and ONNX are generated.
    """
    report = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            original = module.weight.detach().clone()
            integer = torch.round(original)
            clipped = integer.clamp(-128, 127)
            overflow = int(((integer < -128) | (integer > 127)).sum())
            rounded = int((integer != original).sum())
            module.weight.copy_(clipped)
            report.append(
                {
                    "name": name,
                    "minimum_before": float(original.min()),
                    "maximum_before": float(original.max()),
                    "overflow_values_clamped": overflow,
                    "noninteger_values_rounded": rounded,
                    "minimum_after": int(clipped.min()),
                    "maximum_after": int(clipped.max()),
                }
            )
    return report


def verify_onnx_int8_convolution_weights(path):
    """Reject an ONNX graph that DORY would have to truncate to one byte."""
    graph = onnx.load(str(path))
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    report = []
    for node in graph.graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            raise RuntimeError("missing convolution weight initializer for %s" % node.name)
        weight = initializers[node.input[1]]
        if not np.array_equal(weight, np.rint(weight)):
            raise RuntimeError("noninteger convolution weights in %s" % node.input[1])
        minimum, maximum = int(weight.min()), int(weight.max())
        if minimum < -128 or maximum > 127:
            raise RuntimeError(
                "signed-int8 convolution weight overflow in %s: [%d, %d]"
                % (node.input[1], minimum, maximum)
            )
        report.append(
            {"name": node.input[1], "minimum": minimum, "maximum": maximum,
             "values": int(weight.size)}
        )
    if not report:
        raise RuntimeError("ONNX graph has no convolution weights")
    return report


def save_integer_fixture(model, integer_input, output, input_hwc):
    golden = []
    hooks = []
    for module in model.modules():
        if module.__class__.__name__ in ("PACT_Act", "PACT_IntegerAct"):
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, value: golden.append(value.detach())
                )
            )
    with torch.no_grad():
        integer_output = model(integer_input)
    for hook in hooks:
        hook.remove()
    np.savetxt(
        str(output / "input.txt"),
        input_hwc.flatten(),
        fmt="%.3f",
        delimiter=",",
        newline=",\n",
    )
    for layer, activation in enumerate(golden):
        values = activation[0]
        if values.ndim == 3:
            values = values.permute(1, 2, 0)
        np.savetxt(
            str(output / ("out_layer%d.txt" % layer)),
            values.numpy().flatten(),
            fmt="%.3f",
            delimiter=",",
            newline=",\n",
        )
    return integer_output, len(golden)


def configure_residual_requantization(model, factor):
    """Use the smallest exact NeMO residual-add intermediate scale.

    Frontnet-compatible deployment must not depend on an extra post-add
    multiplier in the generated PULP-NN layer.  A factor of one asks NeMO to
    choose the smallest power-of-two D that represents the input scale ratio.
    """
    configured = 0
    for module in model.modules():
        if hasattr(module, "requantization_factor"):
            module.requantization_factor = factor
            configured += 1
    return configured


def load_qat_checkpoint(model, checkpoint_path, graph_name):
    if checkpoint_path is None:
        return False
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("graph") != graph_name:
        raise RuntimeError(
            "QAT checkpoint graph mismatch: %r != %r"
            % (checkpoint.get("graph"), graph_name)
        )
    if checkpoint.get("precision_bits") != 8:
        raise RuntimeError("QAT checkpoint is not int8")
    model.load_state_dict(checkpoint["model"])
    return True


def quantize_encoder(model, calibration_paths, parity_paths, output, add_factor,
                     qat_checkpoint=None):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_float = copy.deepcopy(model).eval()
    model = nemo.transform.quantize_pact(
        model, dummy_input=torch.ones(1, *model.input_shape)
    )
    model.change_precision(bits=8)
    model.reset_alpha_weights()
    model.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration_paths), 32):
            model(image_tensor(calibration_paths[start : start + 32]))
    model.unset_statistics_act()
    model.reset_alpha_act()
    qat_loaded = load_qat_checkpoint(model, qat_checkpoint, "encoder")
    residual_adds = configure_residual_requantization(model, add_factor)
    model.qd_stage(eps_in=1.0 / 255.0)
    model.id_stage()
    weight_report = bound_int8_convolution_weights(model)
    integer_input = image_tensor(calibration_paths[:1]) * 255.0
    input_hwc = integer_input[0].permute(1, 2, 0).numpy()
    integer_output, layers = save_integer_fixture(
        model, integer_input, output, input_hwc
    )
    onnx_path = output / "encoder_int.onnx"
    nemo.utils.export_onnx(
        str(onnx_path), model, model, model.input_shape, perm=None
    )
    epsilon = activation_epsilon(model)
    errors = []
    with torch.no_grad():
        for path in parity_paths:
            image = image_tensor([path]) * 255.0
            reference = model_float(image / 255.0)
            decoded = model(image) * epsilon
            errors.append((decoded - reference).abs().numpy().ravel())
    errors = np.concatenate(errors)
    graph = onnx.load(str(onnx_path))
    return model, model_float, {
        "graph": "encoder",
        "onnx": str(onnx_path),
        "operators": sorted(set(node.op_type for node in graph.graph.node)),
        "output_epsilon": epsilon,
        "output_offset": [0.0],
        "parity_feature_mae": float(np.mean(errors)),
        "parity_feature_max": float(np.max(errors)),
        "integer_output_shape": list(integer_output.shape),
        "integer_nonzero": int((integer_output != 0).sum()),
        "integer_layers": layers,
        "residual_adds_configured": residual_adds,
        "residual_requantization_factor": add_factor,
        "qat_checkpoint": str(qat_checkpoint) if qat_loaded else None,
        "parity_images": len(parity_paths),
        "weight_quantization": weight_report,
        "onnx_int8_convolution_weights": verify_onnx_int8_convolution_weights(
            onnx_path
        ),
    }


def quantize_head(
    name,
    model,
    encoder_integer,
    encoder_float,
    encoder_epsilon,
    calibration_paths,
    parity_paths,
    output,
    add_factor,
    qat_checkpoint=None,
):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_float = copy.deepcopy(model).eval()
    learned_bias = model.output_proj[1].bias.detach().numpy().copy()
    spatial_min = np.full(model.output_channels, np.inf, np.float64)
    with torch.no_grad():
        for start in range(0, len(calibration_paths), 32):
            shared = encoder_float(
                image_tensor(calibration_paths[start : start + 32])
            )
            spatial = model.output_proj[0](model.forward_features(shared))
            flat = spatial.permute(1, 0, 2, 3).contiguous().view(
                model.output_channels, -1
            )
            spatial_min = np.minimum(spatial_min, flat.min(1)[0].numpy())
    offset = np.maximum(-spatial_min, 0.0) + 1.0e-4
    with torch.no_grad():
        model.output_proj[1].bias.copy_(
            torch.from_numpy(offset).to(model.output_proj[1].bias)
        )
    model = nemo.transform.quantize_pact(
        model,
        dummy_input=torch.ones(1, *model.input_shape),
    )
    model.change_precision(bits=8)
    model.reset_alpha_weights()
    model.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(calibration_paths), 32):
            shared = encoder_float(
                image_tensor(calibration_paths[start : start + 32])
            )
            model(shared)
    model.unset_statistics_act()
    model.reset_alpha_act()
    qat_loaded = load_qat_checkpoint(model, qat_checkpoint, name)
    residual_adds = configure_residual_requantization(model, add_factor)
    model.qd_stage(eps_in=encoder_epsilon)
    model.id_stage()
    weight_report = bound_int8_convolution_weights(model)
    image = image_tensor(calibration_paths[:1]) * 255.0
    with torch.no_grad():
        integer_shared = encoder_integer(image)
    input_hwc = integer_shared[0].permute(1, 2, 0).numpy()
    integer_output, layers = save_integer_fixture(
        model, integer_shared, output, input_hwc
    )
    onnx_path = output / ("%s_int.onnx" % name)
    nemo.utils.export_onnx(
        str(onnx_path), model, model, model.input_shape, perm=None
    )
    epsilon = float(model.output_proj[-1].eps_out)
    errors, probability_errors, peak_errors = [], [], []
    decoded_outputs, float_outputs = [], []
    with torch.no_grad():
        for path in parity_paths:
            image = image_tensor([path]) * 255.0
            float_shared = encoder_float(image / 255.0)
            reference = model_float.forward_logits(float_shared)[0].numpy()
            raw = model(encoder_integer(image))[0].numpy()
            decoded = (
                raw * epsilon
                - offset[:, None, None]
                + learned_bias[:, None, None]
            )
            decoded_outputs.append(decoded)
            float_outputs.append(reference)
            errors.append(np.abs(decoded - reference).ravel())
            probability_errors.append(
                np.abs(
                    1.0 / (1.0 + np.exp(-np.clip(decoded, -30.0, 30.0)))
                    - 1.0
                    / (1.0 + np.exp(-np.clip(reference, -30.0, 30.0)))
                ).ravel()
            )
            if name == "corner_head":
                for channel in range(4):
                    ref_y, ref_x = np.unravel_index(
                        np.argmax(reference[channel]), reference[channel].shape
                    )
                    int_y, int_x = np.unravel_index(
                        np.argmax(decoded[channel]), decoded[channel].shape
                    )
                    peak_errors.append(
                        ((ref_x - int_x) ** 2 + (ref_y - int_y) ** 2) ** 0.5
                    )
    np.savez_compressed(
        str(output / ("%s_parity_predictions.npz" % name)),
        paths=np.asarray([str(path) for path in parity_paths]),
        integer_logits=np.asarray(decoded_outputs, dtype=np.float32),
        float_logits=np.asarray(float_outputs, dtype=np.float32),
    )
    errors = np.concatenate(errors)
    probability_errors = np.concatenate(probability_errors)
    graph = onnx.load(str(onnx_path))
    return {
        "graph": name,
        "onnx": str(onnx_path),
        "operators": sorted(set(node.op_type for node in graph.graph.node)),
        "input_epsilon": encoder_epsilon,
        "output_epsilon": epsilon,
        "output_offset": offset.tolist(),
        "learned_bias": learned_bias.tolist(),
        "parity_logit_mae": float(errors.mean()),
        "parity_logit_max": float(errors.max()),
        "parity_probability_mae": float(probability_errors.mean()),
        "parity_probability_max": float(probability_errors.max()),
        "parity_corner_peak_mean_heatmap_px": (
            float(np.mean(peak_errors)) if peak_errors else None
        ),
        "parity_corner_peak_p95_heatmap_px": (
            float(np.percentile(peak_errors, 95)) if peak_errors else None
        ),
        "integer_output_shape": list(integer_output.shape),
        "integer_nonzero": int((integer_output != 0).sum()),
        "integer_layers": layers,
        "residual_adds_configured": residual_adds,
        "residual_requantization_factor": add_factor,
        "qat_checkpoint": str(qat_checkpoint) if qat_loaded else None,
        "parity_images": len(parity_paths),
        "weight_quantization": weight_report,
        "onnx_int8_convolution_weights": verify_onnx_int8_convolution_weights(
            onnx_path
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--parity-images", type=int, default=512)
    parser.add_argument(
        "--residual-requantization-factor", type=int, default=1,
        help="NeMO PACT_IntegerAdd minimum requantization factor; canonical releases use 1.",
    )
    parser.add_argument("--encoder-qat-checkpoint", type=Path)
    parser.add_argument("--corner-head-qat-checkpoint", type=Path)
    parser.add_argument("--danger-head-qat-checkpoint", type=Path)
    args = parser.parse_args()
    paths = sorted(args.dataset.glob("shard_*/hm01b0_mono_*.png"))
    indices = np.linspace(
        0, len(paths) - 1, min(args.calibration_images, len(paths))
    ).astype(int)
    calibration_paths = [paths[index] for index in indices]
    split = json.loads(args.split_file.read_text())
    test_paths = []
    for shard in split["test"]:
        test_paths.extend(
            sorted((args.dataset / shard).glob("hm01b0_mono_*.png"))
        )
    parity_indices = np.linspace(
        0, len(test_paths) - 1, min(args.parity_images, len(test_paths))
    ).astype(int)
    parity_paths = [test_paths[index] for index in parity_indices]
    encoder = EncoderNet()
    load_archive(encoder, args.bridge / "encoder_float_state.npz")
    encoder_integer, encoder_float, encoder_report = quantize_encoder(
        encoder,
        calibration_paths,
        parity_paths,
        args.output / "encoder",
        args.residual_requantization_factor,
        args.encoder_qat_checkpoint,
    )
    reports = [encoder_report]
    for name, model in (
        ("corner_head", CornerHeadNet()),
        ("danger_head", DangerHeadNet()),
    ):
        load_head(model, args.bridge / ("%s_float_state.npz" % name), name)
        reports.append(
            quantize_head(
                name,
                model,
                encoder_integer,
                encoder_float,
                encoder_report["output_epsilon"],
                calibration_paths,
                parity_paths,
                args.output / name,
                args.residual_requantization_factor,
                (args.corner_head_qat_checkpoint if name == "corner_head"
                 else args.danger_head_qat_checkpoint),
            )
        )
    report = {
        "architecture": "Gap8STDCSharedDoryNet",
        "graphs": reports,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nemo_stdc_shared_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
