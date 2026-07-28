#!/usr/bin/env python
"""Python-3.7-compatible NEMO integer export for the packed DORY graph."""

from __future__ import print_function

import argparse
import json
import os
import os
from pathlib import Path

import cv2
import nemo
import numpy as np
import onnx
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                cin, cout, kernel, stride=stride, padding=kernel // 2,
                groups=groups, bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),
        )


class DSConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(DSConv, self).__init__()
        self.depthwise = ConvBNReLU(cin, cin, 3, stride, groups=cin)
        self.pointwise = ConvBNReLU(cin, cout, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualDS(nn.Module):
    def __init__(self, channels):
        super(ResidualDS, self).__init__()
        self.block = DSConv(channels, channels)
        self.add = nemo.quant.pact.PACT_IntegerAdd()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.add(x, self.block(x)))


class PackedDoryNet(nn.Module):
    def __init__(self, packed_channels):
        super(PackedDoryNet, self).__init__()
        self.stem = nn.Sequential(ConvBNReLU(1, 8, 3, 2), DSConv(8, 12))
        self.e1_down = DSConv(12, 20, 2)
        self.e1_refine = ResidualDS(20)
        self.geometry40 = nn.Sequential(
            ConvBNReLU(20, 16, 1),
            ResidualDS(16), ResidualDS(16),
            ResidualDS(16), ResidualDS(16),
            ResidualDS(16), ResidualDS(16),
            ResidualDS(16), ResidualDS(16),
            ResidualDS(16), ResidualDS(16),
            ResidualDS(16), ResidualDS(16),
        )
        self.packed_head = nn.Sequential(DSConv(16, 12))
        self.output_proj = ConvBNReLU(12, packed_channels, kernel=1)
        # DORY's GAP8 convolution kernels emit uint8 activations, not an
        # unquantized int32 terminal tensor.  The per-channel bias shift below
        # makes signed logits nonnegative before this deploy-only quantizer.
        # Represent the trained final Conv bias through BatchNorm beta so old
        # NEMO lowers the boundary as a fused BNReluConvolution. Its plain
        # terminal ReluConvolution integer path collapses this model to zero.

    def forward_features(self, x):
        x = self.stem(x)
        e1 = self.e1_refine(self.e1_down(x))
        return self.packed_head(self.geometry40(e1))

    def forward_logits(self, x):
        features = self.forward_features(x)
        return self.output_proj[1](self.output_proj[0](features))

    def forward(self, x):
        return self.output_proj(self.forward_features(x))


def calibration_images(dataset, count):
    paths = sorted(Path(dataset).glob("shard_*/hm01b0_mono_*.png"))
    if not paths:
        raise RuntimeError("no calibration HM01B0 images under %s" % dataset)
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).astype(int)
    return [paths[i] for i in indices]


def render_integer_prediction(image, decoded, source_name, output_path):
    """Render the exact NEMO integer-stage output after affine decoding."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    corners = base.copy()
    corner_colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)
    ]
    corner_names = ["TL", "TR", "BR", "BL"]
    for channel in range(4):
        y, x = np.unravel_index(
            np.argmax(decoded[channel]), decoded[channel].shape
        )
        px = int(round((float(x) + 0.5) * 4.0))
        py = int(round((float(y) + 0.5) * 4.0))
        cv2.circle(corners, (px, py), 4, corner_colors[channel], 1)
        cv2.putText(
            corners, corner_names[channel], (px + 3, max(10, py - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, corner_colors[channel], 1,
            cv2.LINE_AA,
        )

    danger_logits = decoded[4].reshape(20, 2, 20, 2).mean((1, 3))
    danger = 1.0 / (1.0 + np.exp(-np.clip(danger_logits, -30.0, 30.0)))
    danger_u8 = np.uint8(np.clip(danger * 255.0, 0.0, 255.0))
    danger_color = cv2.applyColorMap(
        cv2.resize(danger_u8, (160, 160), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_JET,
    )
    danger_panel = cv2.addWeighted(base, 0.45, danger_color, 0.55, 0.0)
    danger_scale = cv2.applyColorMap(
        (np.arange(160, dtype=np.uint16).reshape(1, 160) * 255 // 159)
        .astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    danger_panel[152:160] = cv2.resize(
        danger_scale, (160, 8), interpolation=cv2.INTER_NEAREST
    )

    range_logits = decoded[5].reshape(20, 2, 20, 2).mean((1, 3))
    inverse_range = 1.0 / (
        1.0 + np.exp(-np.clip(range_logits, -30.0, 30.0))
    )
    range_u8 = np.uint8(np.clip(inverse_range * 255.0, 0.0, 255.0))
    range_color = cv2.applyColorMap(
        cv2.resize(range_u8, (160, 160), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_TURBO,
    )
    range_panel = cv2.addWeighted(base, 0.45, range_color, 0.55, 0.0)

    gate_panel = base.copy()
    if decoded.shape[0] >= 8:
        gate = 1.0 / (
            1.0 + np.exp(-np.clip(decoded[7], -30.0, 30.0))
        )
        gate_mask = cv2.resize(
            np.uint8(gate >= 0.5), (160, 160),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        gate_panel[gate_mask] = (
            0.35 * gate_panel[gate_mask] + 0.65 * np.array([0, 255, 0])
        ).astype(np.uint8)

    panels = [corners, gate_panel, danger_panel, range_panel]
    labels = [
        "integer corners", "integer gate",
        "hazard: blue=low red=high", "inverse range",
    ]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (159, 15), (0, 0, 0), -1)
        cv2.putText(
            panel, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    montage = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    cv2.putText(
        montage, source_name[:48], (3, 317), cv2.FONT_HERSHEY_SIMPLEX,
        0.3, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError("failed to write %s" % output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nemo-qat-checkpoint", type=Path)
    parser.add_argument("--parity-images", type=int, default=32)
    parser.add_argument(
        "--parity-output-dir", type=Path,
        help="Write exact integer-stage prediction montages and NPZ tensors.",
    )
    parser.add_argument("--requantization-factor", type=int, default=32)
    parser.add_argument(
        "--output-equalization-limit", type=float, default=1.0,
        help=(
            "Maximum per-channel amplification before the shared uint8 output "
            "quantizer; 1 disables terminal-channel equalization."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Old NEMO/PyTorch CPU kernels otherwise permit thread-scheduling-dependent
    # calibration results. Export is infrequent, so deterministic single-thread
    # execution is preferable to a small speedup.
    torch.set_num_threads(1)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    if torch.cuda.is_available():
        raise RuntimeError(
            "CUDA must be hidden: old NEMO otherwise allocates parameters on "
            "an unallocated cluster GPU"
        )

    bridge_report = json.loads((args.bridge / "bridge_report.json").read_text())
    model = PackedDoryNet(bridge_report["packed_channels"]).eval()
    archive = np.load(str(args.bridge / "packed_float_state.npz"))
    legacy_state = model.state_dict()
    deploy_only_state = {
        key for key in legacy_state if key.startswith("output_proj.")
    }
    remapped_archive_keys = {
        "packed_head.1.weight": "output_proj.0.weight",
    }
    shared_archive_keys = {
        key for key in archive.files if not key.startswith("packed_head.1.")
    }
    missing = sorted(
        set(legacy_state)
        - shared_archive_keys
        - set(remapped_archive_keys.values())
        - deploy_only_state
    )
    unexpected = sorted(
        set(archive.files)
        - shared_archive_keys
        - set(remapped_archive_keys)
        - {"packed_head.1.bias"}
    )
    if missing or unexpected:
        raise RuntimeError("state mismatch missing=%s unexpected=%s" % (missing, unexpected))
    legacy_state.update({
        key: torch.from_numpy(archive[key]) for key in shared_archive_keys
    })
    legacy_state["output_proj.0.weight"] = torch.from_numpy(
        archive["packed_head.1.weight"]
    )
    # Make eval-mode BatchNorm exactly y=x+bias before deploy calibration.
    bn_eps = model.output_proj[1].eps
    legacy_state["output_proj.1.weight"] = torch.full_like(
        legacy_state["output_proj.1.weight"], (1.0 + bn_eps) ** 0.5
    )
    legacy_state["output_proj.1.bias"] = torch.from_numpy(
        archive["packed_head.1.bias"]
    )
    legacy_state["output_proj.1.running_mean"].zero_()
    legacy_state["output_proj.1.running_var"].fill_(1.0)
    model.load_state_dict(legacy_state)

    paths = calibration_images(args.dataset, args.calibration_images)
    # Preserve signed logits through an affine uint8 boundary.  This is
    # deploy-only: training and loss computation continue to use raw logits.
    channel_min = np.full(bridge_report["packed_channels"], np.inf, np.float64)
    channel_max = np.full(bridge_report["packed_channels"], -np.inf, np.float64)
    spatial_min = np.full(bridge_report["packed_channels"], np.inf, np.float64)
    spatial_max = np.full(bridge_report["packed_channels"], -np.inf, np.float64)
    learned_output_bias = model.output_proj[1].bias.detach().cpu().numpy().copy()
    parity_references = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            images = [
                cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                for path in paths[start : start + args.batch_size]
            ]
            tensor = torch.from_numpy(np.asarray(images)).unsqueeze(1).float() / 255.0
            logits = model.forward_logits(tensor)
            remaining = args.parity_images - len(parity_references)
            if remaining > 0:
                parity_references.extend(
                    item.detach().cpu().numpy().copy()
                    for item in logits[:remaining]
                )
            features = model.forward_features(tensor)
            spatial_logits = model.output_proj[0](features)
            flat_logits = logits.permute(1, 0, 2, 3).contiguous().view(
                logits.shape[1], -1
            )
            flat_spatial = spatial_logits.permute(1, 0, 2, 3).contiguous().view(
                spatial_logits.shape[1], -1
            )
            channel_min = np.minimum(
                channel_min, flat_logits.min(dim=1)[0].cpu().numpy()
            )
            channel_max = np.maximum(
                channel_max, flat_logits.max(dim=1)[0].cpu().numpy()
            )
            spatial_min = np.minimum(
                spatial_min, flat_spatial.min(dim=1)[0].cpu().numpy()
            )
            spatial_max = np.maximum(
                spatial_max, flat_spatial.max(dim=1)[0].cpu().numpy()
            )
    # NEMO exposes one epsilon for the packed terminal tensor. Without
    # equalization, the wide danger/gate logit ranges consume nearly all 8-bit
    # levels and the four narrow corner channels lose their peaks. Amplify
    # each terminal convolution channel to use a comparable physical range,
    # then undo that scale in the controller-facing affine decoder.
    spatial_span = np.maximum(spatial_max - spatial_min, 1.0e-6)
    output_channel_scale = np.clip(
        float(np.max(spatial_span)) / spatial_span,
        1.0,
        max(float(args.output_equalization_limit), 1.0),
    )
    with torch.no_grad():
        # Apply the scale in the fused terminal BatchNorm, not to the
        # convolution weights. Old NEMO quantizes the convolution tensor with
        # one weight scale, so changing individual output filters there would
        # damage the unscaled danger channel.
        model.output_proj[1].weight.mul_(
            torch.from_numpy(output_channel_scale)
            .to(model.output_proj[1].weight)
        )
    physical_spatial_min = spatial_min * output_channel_scale
    physical_spatial_max = spatial_max * output_channel_scale
    output_offset = np.maximum(-physical_spatial_min, 0.0) + 1.0e-4
    output_bn = model.output_proj[1]
    with torch.no_grad():
        output_bn.bias.copy_(torch.from_numpy(output_offset).to(output_bn.bias))

    model = nemo.transform.quantize_pact(
        model, dummy_input=torch.ones(1, 1, 160, 160)
    )
    model.change_precision(bits=8)
    model.reset_alpha_weights()
    model.set_statistics_act()
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            images = [
                cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                for path in paths[start : start + args.batch_size]
            ]
            tensor = torch.from_numpy(np.asarray(images)).unsqueeze(1).float() / 255.0
            model(tensor)
    model.unset_statistics_act()
    model.reset_alpha_act()
    if args.nemo_qat_checkpoint:
        qat_state = torch.load(
            str(args.nemo_qat_checkpoint), map_location="cpu"
        )
        if qat_state.get("architecture") != bridge_report["architecture"]:
            raise RuntimeError(
                "NEMO-QAT architecture mismatch: %r != %r"
                % (
                    qat_state.get("architecture"),
                    bridge_report["architecture"],
                )
            )
        if not np.allclose(
            np.asarray(qat_state["output_offset"]), output_offset,
            rtol=1.0e-5, atol=1.0e-5,
        ):
            raise RuntimeError("NEMO-QAT output offset mismatch")
        model.load_state_dict(qat_state["model"])
        model.eval()
    for module in model.modules():
        if hasattr(module, "requantization_factor"):
            module.requantization_factor = args.requantization_factor
    with torch.no_grad():
        fq_output = model(
            torch.from_numpy(
                cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
            ).unsqueeze(0).unsqueeze(0).float() / 255.0
        )

    model.qd_stage(eps_in=1.0 / 255.0)
    with torch.no_grad():
        qd_output = model(
            torch.from_numpy(
                cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
            ).unsqueeze(0).unsqueeze(0).float() / 255.0
        )
    model.id_stage()
    golden_activations = []
    hooks = []

    def capture(module, inputs, output):
        golden_activations.append(output.detach().cpu())

    for module in model.modules():
        if module.__class__.__name__ in ("PACT_Act", "PACT_IntegerAct"):
            hooks.append(module.register_forward_hook(capture))
    input_image = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    integer_input = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        final_integer_output = model(integer_input).detach().cpu()
    final_integer_nonzero = int((final_integer_output != 0).sum().item())
    if final_integer_nonzero == 0:
        tail_ranges = [
            (
                module_index,
                float(activation.min()),
                float(activation.max()),
                int((activation != 0).sum().item()),
            )
            for module_index, activation in enumerate(golden_activations[-4:])
        ]
        output_parameter_ranges = {
            name: (
                float(parameter.min()),
                float(parameter.max()),
                int((parameter != 0).sum().item()),
            )
            for name, parameter in model.named_parameters()
            if name.startswith("output_proj")
        }
        raise RuntimeError(
            "integer-deployable terminal tensor is entirely zero; refusing "
            "to export a degenerate GAP8 network; fake-quant range=%r, "
            "quantized-deployable range=%r, last activation ranges=%r, "
            "output parameters=%r"
            % (
                (float(fq_output.min()), float(fq_output.max())),
                (float(qd_output.min()), float(qd_output.max())),
                tail_ranges,
                output_parameter_ranges,
            )
        )
    expected_golden_activations = len(hooks)
    for hook in hooks:
        hook.remove()
    if len(golden_activations) != expected_golden_activations:
        raise RuntimeError(
            "expected %d fused GAP8 layer outputs, captured %d"
            % (expected_golden_activations, len(golden_activations))
        )
    for layer, activation in enumerate(golden_activations):
        values = activation[0]
        if values.ndim == 3:
            values = values.permute(1, 2, 0)
        np.savetxt(
            str(args.output / ("out_layer%d.txt" % layer)),
            values.numpy().flatten(),
            header="NEMO integer activation shape %s" % list(values.shape),
            fmt="%.3f", delimiter=",", newline=",\n",
        )
    onnx_path = args.output / "gap8_packed_int.onnx"
    nemo.utils.export_onnx(
        str(onnx_path), model, model, (1, 160, 160), perm=None
    )
    np.savetxt(
        str(args.output / "input.txt"),
        input_image.flatten(),
        header="HM01B0 uint8 input shape [160,160,1]",
        fmt="%d", delimiter=",", newline=",\n",
    )
    graph = onnx.load(str(onnx_path))
    final_quantizer = model.output_proj[-1]
    output_epsilon = float(final_quantizer.eps_out)
    parity_errors = []
    parity_corner_peak_error = []
    parity_danger_probability_error = []
    parity_gate_probability_error = []
    if args.parity_output_dir:
        args.parity_output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, reference in enumerate(parity_references):
            image = cv2.imread(
                str(paths[index]), cv2.IMREAD_GRAYSCALE
            )
            raw = model(
                torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            )[0].detach().cpu().numpy()
            decoded = (
                (
                    raw * output_epsilon
                    - output_offset[:, None, None]
                )
                / output_channel_scale[:, None, None]
                + learned_output_bias[:, None, None]
            )
            if args.parity_output_dir:
                sample_stem = "integer_prediction_%04d" % index
                np.savez_compressed(
                    str(args.parity_output_dir / (sample_stem + ".npz")),
                    source_path=str(paths[index]),
                    integer_output=raw,
                    decoded_logits=decoded,
                    float_logits=reference,
                )
                render_integer_prediction(
                    image, decoded, paths[index].name,
                    args.parity_output_dir / (sample_stem + ".png"),
                )
            parity_errors.append(np.abs(decoded - reference).ravel())
            for channel in range(4):
                ref_y, ref_x = np.unravel_index(
                    np.argmax(reference[channel]), reference[channel].shape
                )
                int_y, int_x = np.unravel_index(
                    np.argmax(decoded[channel]), decoded[channel].shape
                )
                parity_corner_peak_error.append(
                    ((ref_x - int_x) ** 2 + (ref_y - int_y) ** 2) ** 0.5
                )
            ref_danger = 1.0 / (
                1.0 + np.exp(-reference[4].reshape(20, 2, 20, 2).mean((1, 3)))
            )
            int_danger = 1.0 / (
                1.0 + np.exp(-decoded[4].reshape(20, 2, 20, 2).mean((1, 3)))
            )
            parity_danger_probability_error.append(
                np.abs(ref_danger - int_danger).ravel()
            )
            if bridge_report["packed_channels"] == 8:
                ref_gate = 1.0 / (1.0 + np.exp(-reference[7]))
                int_gate = 1.0 / (1.0 + np.exp(-decoded[7]))
                parity_gate_probability_error.append(
                    np.abs(ref_gate - int_gate).ravel()
                )
    parity_errors = np.concatenate(parity_errors)
    parity_corner_peak_error = np.asarray(parity_corner_peak_error)
    parity_danger_probability_error = np.concatenate(
        parity_danger_probability_error
    )
    parity_gate_probability_error = (
        np.concatenate(parity_gate_probability_error)
        if parity_gate_probability_error else np.asarray([])
    )
    report = {
        "architecture": bridge_report["architecture"],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "packed_channels": bridge_report["packed_channels"],
        "calibration_images": len(paths),
        "calibration_first": str(paths[0]),
        "fake_quant_output_shape": list(fq_output.shape),
        "fake_quant_output_range": [
            float(fq_output.min()), float(fq_output.max())
        ],
        "quantized_deployable_output_range": [
            float(qd_output.min()), float(qd_output.max())
        ],
        "onnx": str(onnx_path),
        "onnx_operators": sorted(set(node.op_type for node in graph.graph.node)),
        "nemo_stages": ["fake_quant", "quantized_deployable", "integer_deployable"],
        "nemo_qat_checkpoint": (
            str(args.nemo_qat_checkpoint)
            if args.nemo_qat_checkpoint else None
        ),
        "requantization_factor": args.requantization_factor,
        "input_epsilon": 1.0 / 255.0,
        "packed_output_quantization": {
            "dtype": "uint8",
            "epsilon": output_epsilon,
            "per_channel_spatial_offset": output_offset.tolist(),
            "per_channel_output_scale": output_channel_scale.tolist(),
            "per_channel_learned_bias": learned_output_bias.tolist(),
            "dequantization": (
                "logit[channel] = (uint8_value * epsilon "
                "- per_channel_spatial_offset[channel]) "
                "/ per_channel_output_scale[channel] "
                "+ per_channel_learned_bias[channel]"
            ),
            "calibration_logit_min": channel_min.tolist(),
            "calibration_logit_max": channel_max.tolist(),
            "calibration_spatial_min": spatial_min.tolist(),
            "calibration_spatial_max": spatial_max.tolist(),
            "physical_spatial_min_after_equalization": (
                physical_spatial_min.tolist()
            ),
            "physical_spatial_max_after_equalization": (
                physical_spatial_max.tolist()
            ),
        },
        "golden_activation_files": len(golden_activations),
        "final_integer_output_shape": list(final_integer_output.shape),
        "final_integer_output_range": [
            float(final_integer_output.min()), float(final_integer_output.max())
        ],
        "final_integer_output_nonzero": final_integer_nonzero,
        "integer_parity": {
            "images": len(parity_references),
            "logit_mae": float(parity_errors.mean()),
            "logit_p95_absolute_error": float(
                np.percentile(parity_errors, 95)
            ),
            "corner_peak_mean_error_heatmap_px": float(
                parity_corner_peak_error.mean()
            ),
            "corner_peak_p95_error_heatmap_px": float(
                np.percentile(parity_corner_peak_error, 95)
            ),
            "danger_probability_mae": float(
                parity_danger_probability_error.mean()
            ),
            "gate_probability_mae": (
                float(parity_gate_probability_error.mean())
                if len(parity_gate_probability_error) else None
            ),
        },
    }
    (args.output / "nemo_export_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
