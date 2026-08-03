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

## Live NanoCockpit Wi-Fi inference

`live_onnx.py` runs this same ONNX and decoder on NanoCockpit's CPX stream;
it is not compatible with Bitcraze's `wifi-img-streamer`. Flash NanoCockpit's
streaming-only GAP8 image (`STDC_STREAM_ONLY=1`) and its NINA CPX bridge, then
connect the laptop to the AI-deck Wi-Fi AP (or use the deck's station-mode
address). That firmware streams the required `160x120` monochrome center crop.
With the NanoCockpit checkout adjacent to this repository, run:

```bash
python live_onnx.py
```

### GAP8 image switching

From `tinympc-nanocockpit/src/gap`, the following command builds and flashes
the diagnostic image used by `live_onnx.py`:

```bash
./gap8.sh examples/tiny-racer clean all STDC_STREAM_ONLY=1
```

This image streams camera crops only: it does **not** run the GAP8 network and
does **not** transmit the STM32 `0x38` sequential-obstacle UART packet. Before
returning to a flight test, always restore the onboard sequential image:

```bash
./gap8.sh examples/tiny-racer clean all \
  NETWORK_NAME=gap8-sequential-bothflights-qat-dory
```

The restored image publishes one CRC-protected 48-byte `0x38` packet after
each inference: capture timestamp, sequence, gate flag, four clearances, and
four confidence scores. The NINA CPX bridge is required for diagnostics but
does not need to be reflashed when it is already serving the CPX streamer.

The default endpoint is the AI-deck AP at `192.168.4.1:5000`. For station mode
or a non-default port, use `--host <ip> --port <port>`. If NanoCockpit is in a
different location, pass `--nanocockpit-root /path/to/tinympc-nanocockpit` (or
set `NANOCOCKPIT_ROOT`). It opens a scaled two-panel preview: the full camera
frame and a direction panel with the four fixed angles, clearance, confidence,
and the controller's open/blocked classification. The default is the flight
policy (`--safe-min 0.32`, `--confidence-min 0.0`); change those options to
inspect a proposed threshold. Use `--display-scale 2` for a smaller window.
Press `q` or Escape to quit. Use `--no-display --jsonl` for headless JSONL
output, `--max-frames N` for a bounded run, and only use `--resize-to-model`
when you intentionally accept the changed camera geometry. The normal CPX
acknowledgement is retained for stream timing, but no inference output or
flight command is sent back.

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
