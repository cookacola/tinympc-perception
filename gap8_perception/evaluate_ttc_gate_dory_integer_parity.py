#!/usr/bin/env python3
"""Summarize decoded float-versus-NeMO parity for DORY gate/TTC heads.

The input archives are written by :mod:`nemo_ttc_gate_dory_export` after its
integer terminal values have been decoded back to the float-logit domain.  No
model framework is needed here: this is deliberately a NumPy-only acceptance
check of the deployment-visible decoders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GATE_CHANNELS = 8
TTC_CHANNELS = 7


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _softplus(values: np.ndarray) -> np.ndarray:
    """Stable NumPy equivalent of torch.nn.functional.softplus(beta=1)."""
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def _require_logits(name: str, values: np.ndarray, channels: int) -> None:
    if values.ndim != 4 or values.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape [N,{channels},H,W], received {values.shape}"
        )
    if values.shape[0] == 0 or values.shape[2:] != (20, 20):
        raise ValueError(f"{name} must have nonempty 20x20 predictions, received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def _gate_metrics(float_logits: np.ndarray, integer_logits: np.ndarray) -> dict:
    _require_logits("float gate logits", float_logits, GATE_CHANNELS)
    _require_logits("integer gate logits", integer_logits, GATE_CHANNELS)
    if float_logits.shape != integer_logits.shape:
        raise ValueError("float and integer gate logits have different shapes")

    samples = float_logits.shape[0]
    float_peaks = float_logits[:, :4].reshape(samples, 4, -1).argmax(axis=-1)
    integer_peaks = integer_logits[:, :4].reshape(samples, 4, -1).argmax(axis=-1)
    float_y, float_x = np.divmod(float_peaks, 20)
    integer_y, integer_x = np.divmod(integer_peaks, 20)
    displacement = np.hypot(float_x - integer_x, float_y - integer_y)

    float_visibility_logits = float_logits[:, 4:8].mean(axis=(-2, -1))
    integer_visibility_logits = integer_logits[:, 4:8].mean(axis=(-2, -1))
    float_probability = _sigmoid(float_visibility_logits)
    integer_probability = _sigmoid(integer_visibility_logits)
    float_visible = float_probability >= 0.5
    integer_visible = integer_probability >= 0.5
    return {
        "heatmap_peak_cell_exact_agreement": float((float_peaks == integer_peaks).mean()),
        "heatmap_peak_cell_exact_count": int((float_peaks == integer_peaks).sum()),
        "heatmap_peak_cell_total": int(float_peaks.size),
        "heatmap_peak_cell_displacement_mean": float(displacement.mean()),
        "heatmap_peak_cell_displacement_p95": float(np.percentile(displacement, 95)),
        "visibility_probability_mae": float(np.abs(float_probability - integer_probability).mean()),
        "visibility_threshold": 0.5,
        "visibility_thresholded_agreement": float((float_visible == integer_visible).mean()),
        "visibility_thresholded_agreement_count": int((float_visible == integer_visible).sum()),
        "visibility_thresholded_total": int(float_visible.size),
    }


def _ttc_metrics(float_logits: np.ndarray, integer_logits: np.ndarray) -> dict:
    _require_logits("float TTC logits", float_logits, TTC_CHANNELS)
    _require_logits("integer TTC logits", integer_logits, TTC_CHANNELS)
    if float_logits.shape != integer_logits.shape:
        raise ValueError("float and integer TTC logits have different shapes")

    float_inverse_ttc, integer_inverse_ttc = _softplus(float_logits[:, 0]), _softplus(integer_logits[:, 0])
    float_inverse_depth, integer_inverse_depth = _softplus(float_logits[:, 1]), _softplus(integer_logits[:, 1])
    flow_error = integer_logits[:, 2:4] - float_logits[:, 2:4]
    flow_epe = np.sqrt(np.square(flow_error).sum(axis=1))
    float_risk = _softmax(float_logits[:, 4:7], axis=1)
    integer_risk = _softmax(integer_logits[:, 4:7], axis=1)
    return {
        "inverse_ttc_softplus_mae": float(np.abs(integer_inverse_ttc - float_inverse_ttc).mean()),
        "inverse_depth_softplus_mae": float(np.abs(integer_inverse_depth - float_inverse_depth).mean()),
        "flow_mae": float(np.abs(flow_error).mean()),
        "flow_epe_mean": float(flow_epe.mean()),
        "risk_softmax_class_agreement": float(
            (float_risk.argmax(axis=1) == integer_risk.argmax(axis=1)).mean()
        ),
        "risk_critical_probability_mae": float(
            np.abs(float_risk[:, 2] - integer_risk[:, 2]).mean()
        ),
    }


def parity_metrics(
    gate_float_logits: np.ndarray,
    gate_integer_logits: np.ndarray,
    ttc_float_logits: np.ndarray,
    ttc_integer_logits: np.ndarray,
) -> dict:
    """Compute all deployment-visible parity metrics from decoded logit tensors."""
    return {
        "samples": int(gate_float_logits.shape[0]),
        "gate": _gate_metrics(gate_float_logits, gate_integer_logits),
        "ttc": _ttc_metrics(ttc_float_logits, ttc_integer_logits),
    }


def _archive(path: Path, expected_channels: int, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"paths", "float_logits", "integer_logits"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing keys {sorted(missing)}")
        paths = archive["paths"]
        float_logits = archive["float_logits"].astype(np.float64, copy=False)
        integer_logits = archive["integer_logits"].astype(np.float64, copy=False)
    _require_logits(f"{name} float logits", float_logits, expected_channels)
    _require_logits(f"{name} integer logits", integer_logits, expected_channels)
    if paths.shape != (float_logits.shape[0],):
        raise ValueError(f"{path} paths do not align with predictions")
    return paths, float_logits, integer_logits


def evaluate_archives(integer_dir: str | Path) -> dict:
    integer_dir = Path(integer_dir)
    gate_paths, gate_float, gate_integer = _archive(
        integer_dir / "gate_head" / "gate_head_parity_predictions.npz", GATE_CHANNELS, "gate"
    )
    ttc_paths, ttc_float, ttc_integer = _archive(
        integer_dir / "ttc_head" / "ttc_head_parity_predictions.npz", TTC_CHANNELS, "TTC"
    )
    if not np.array_equal(gate_paths, ttc_paths):
        raise ValueError("gate_head and ttc_head parity archives have different path ordering")
    if gate_float.shape[0] != ttc_float.shape[0]:
        raise ValueError("gate_head and ttc_head parity archives have different sample counts")
    return parity_metrics(gate_float, gate_integer, ttc_float, ttc_integer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-dir", type=Path, required=True,
                        help="Directory containing gate_head/ and ttc_head/ parity archives.")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON output path; metrics are always printed.")
    args = parser.parse_args()
    report = evaluate_archives(args.integer_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
