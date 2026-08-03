#!/usr/bin/env python3
"""Run the deployed ONNX model on NanoCockpit's live AI-deck CPX stream.

The NanoCockpit streaming-only GAP8 firmware sends raw monochrome camera
frames through its NINA CPX bridge. This script runs the laptop ONNX decoder
on those frames; it never sends flight or neural-network outputs to the deck.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from run_onnx import ROOT, annotate, decode, preprocess_image


def nanocockpit_client_path(requested_root: Path | None) -> Path:
    """Find NanoCockpit's maintained CPX Python client without installing it."""
    repository_root = Path(__file__).resolve().parents[2]
    candidates = []
    if requested_root is not None:
        candidates.append(requested_root)
    if configured_root := os.environ.get("NANOCOCKPIT_ROOT"):
        candidates.append(Path(configured_root))
    candidates.append(repository_root.parent / "tinympc-nanocockpit")

    for root in candidates:
        client = root.expanduser().resolve() / "src/client/aideck_cpx_streamer"
        if (client / "aideck_cpx_streamer/cpx/__init__.py").is_file():
            return client
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "could not find NanoCockpit's aideck_cpx_streamer client; pass "
        f"--nanocockpit-root or set NANOCOCKPIT_ROOT (searched: {searched})"
    )


def load_streamer_client(nanocockpit_root: Path | None):
    client_path = nanocockpit_client_path(nanocockpit_root)
    if str(client_path) not in sys.path:
        sys.path.insert(0, str(client_path))
    from aideck_cpx_streamer.cpx import StreamerClient

    return StreamerClient


def prepare_stream_frame(image: np.ndarray, resize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Apply the deployed model's sensor-frame/crop contract to a live image."""
    if image.ndim != 2:
        raise ValueError(f"expected a monochrome NanoCockpit stream, got image shape {image.shape}")
    if image.shape not in ((160, 160), (120, 160)):
        if not resize:
            raise ValueError(
                f"stream delivered {image.shape[1]}x{image.shape[0]}; model requires 160x160 "
                "or NanoCockpit's 160x120 center crop. Configure the streaming-only "
                "GAP8 firmware, or explicitly use --resize-to-model."
            )
        image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    return preprocess_image(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--host", default="192.168.4.1", help="AI-deck Wi-Fi IP or hostname")
    parser.add_argument("-p", "--port", type=int, default=5000, help="NanoCockpit CPX TCP port")
    parser.add_argument("--nanocockpit-root", type=Path, help="path to the NanoCockpit checkout")
    parser.add_argument("--model", type=Path, default=ROOT / "sequential_int.onnx")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="do not open an OpenCV window")
    parser.add_argument("--display-scale", type=float, default=3.0,
                        help="scale the diagnostic window (default: %(default)s)")
    parser.add_argument("--safe-min", type=float, default=0.32,
                        help="clearance threshold shown for an open direction")
    parser.add_argument("--confidence-min", type=float, default=0.0,
                        help="confidence threshold shown for an open direction")
    parser.add_argument("--jsonl", action="store_true", help="also print one JSON prediction per frame")
    parser.add_argument("--no-udp-send", action="store_false", dest="udp_send",
                        help="send acknowledgements through TCP rather than UDP")
    parser.add_argument("--resize-to-model", action="store_true", help="resize unexpected input to 160x160")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.display_scale <= 0:
        parser.error("--display-scale must be positive")

    StreamerClient = load_streamer_client(args.nanocockpit_root)
    manifest = json.loads((ROOT / "quantization_manifest.json").read_text())
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    frame_count = 0
    started = time.monotonic()
    client = StreamerClient(host=args.host, port=args.port, udp_send=args.udp_send)
    # StreamerClient installs a shutdown-only SIGINT handler. Restore Python's
    # normal handler so Ctrl+C raises KeyboardInterrupt, reaches ``finally``,
    # and exits instead of merely issuing repeated shutdown requests.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        for image, _tof_frame, metadata in client.receive():
            # Keep the normal NanoCockpit streaming RTT/flow-control path alive,
            # but deliberately do not send this model's output to the aircraft.
            client.send_reply(metadata, None)
            sensor_frame, tensor = prepare_stream_frame(image, args.resize_to_model)
            raw = session.run(None, {input_name: tensor})[0]
            result = decode(raw, float(manifest["scale"]))
            frame_count += 1
            elapsed = time.monotonic() - started
            result.update(
                frame=frame_count,
                stream_source_size=[int(image.shape[1]), int(image.shape[0])],
                stream_frame_id=int(metadata.frame_id),
                fps=frame_count / elapsed if elapsed else 0.0,
                model_sha256="620fdb49f94abd7adf212b15b0858c49ed46f85f89fdbc4e05d28453c5c9f9b6",
            )
            if args.jsonl:
                print(json.dumps(result, separators=(",", ":")), flush=True)

            if not args.no_display:
                panel = annotate(sensor_frame, result, args.safe_min,
                                 args.confidence_min)
                cv2.putText(panel, f"stream {result['stream_frame_id']}  {result['fps']:.1f} fps",
                            (170, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                            (180, 180, 180), 1, cv2.LINE_AA)
                display = cv2.resize(panel, None, fx=args.display_scale,
                                     fy=args.display_scale,
                                     interpolation=cv2.INTER_NEAREST)
                cv2.imshow("NanoCockpit sequential ONNX", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.shutdown()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
