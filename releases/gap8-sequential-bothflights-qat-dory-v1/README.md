# GAP8 sequential both-flights QAT ONNX bundle

This bundle is the exact quantized ONNX used by the deployed Tiny Racer
network `gap8-sequential-bothflights-qat-dory`, trained with real flight 06
and flight 07 data. It is a CPU-laptop inference package; it does not require
NeMO, PyTorch, DORY, GAP SDK, or a GPU.

The model file is `sequential_int.onnx` and its required SHA-256 is:

```
620fdb49f94abd7adf212b15b0858c49ed46f85f89fdbc4e05d28453c5c9f9b6
```

Install and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_onnx.py /path/to/frame.png --annotated-image prediction.png
```

The input is either a `160x160` HM01B0 frame (the runner uses rows 20–139), or
the already-cropped `160x120` monochrome image. The integer ONNX output is
`uint8 [1,12,15,20]`; the exact dequantization is
`logical_score = uint8 * 0.11307859420776367 - 6.0`.

`run_onnx.py` prints ordered TL/TR/BR/BL corner locations, peak and ambiguity
scores, four fixed-normal clearances, four confidence scores, and whether the
canonical gate-quality checks accept the corner quadrilateral. The channel and
preprocessing contract is recorded in `output_contract.json`.

Only this ONNX is required to run the newly trained model on a laptop. DORY
generated C, activation dumps, and older sequential/STDC ONNX exports are not
runtime dependencies and are intentionally excluded.
