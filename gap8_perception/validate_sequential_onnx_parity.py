#!/usr/bin/env python3
"""Compare a NEMO integer fixture with its exported sequential ONNX graph.

NeMO writes activation fixtures in HWC order for DORY while ONNX uses NCHW.
This gate makes that layout conversion explicit and records integer-LSB error,
so an unnoticed transpose, channel permutation, or output-scale change cannot
be mistaken for a numerically close export.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_csv(path: Path) -> np.ndarray:
    # NeMO terminates each line (including the final line) with a comma.
    return np.fromstring(path.read_text().replace("\n", "").rstrip(","), sep=",", dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integer-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-lsb-error", type=int, default=1)
    args = parser.parse_args()
    root = args.integer_dir
    image = load_csv(root / "input.txt")
    expected_hwc = load_csv(root / "output.txt")
    if image.size != 120 * 160:
        raise ValueError("expected a 160x120 integer fixture")
    if expected_hwc.size != 12 * 15 * 20:
        raise ValueError("expected 12x15x20 output fixture")
    session = ort.InferenceSession(
        str(root / "sequential_int.onnx"), providers=["CPUExecutionProvider"]
    )
    output = session.run(
        None, {session.get_inputs()[0].name: image.reshape(1, 1, 120, 160)}
    )[0]
    if tuple(output.shape) != (1, 12, 15, 20):
        raise ValueError("unexpected ONNX output shape %r" % (output.shape,))
    actual_hwc = output[0].transpose(1, 2, 0).reshape(-1)
    difference = actual_hwc.astype(np.int32) - expected_hwc.astype(np.int32)
    report = {
        "passed": bool(np.max(np.abs(difference)) <= args.max_lsb_error),
        "comparison": "NEMO integer output versus ONNX Runtime output",
        "layout": {"nemo_fixture": "HWC", "onnx": "NCHW"},
        "output_shape": [1, 12, 15, 20],
        "elements": int(difference.size),
        "different_elements": int(np.count_nonzero(difference)),
        "max_absolute_lsb_error": int(np.max(np.abs(difference))),
        "max_allowed_lsb_error": args.max_lsb_error,
        "first_differences": [
            {
                "flat_hwc_index": int(index),
                "y_x_channel": [
                    int(value) for value in np.unravel_index(index, (15, 20, 12))
                ],
                "onnx": int(actual_hwc[index]),
                "nemo": int(expected_hwc[index]),
            }
            for index in np.flatnonzero(difference)[:16]
        ],
    }
    report_path = args.report or root / "onnx_runtime_parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("ONNX Runtime parity exceeds allowed integer LSB error")


if __name__ == "__main__":
    main()
