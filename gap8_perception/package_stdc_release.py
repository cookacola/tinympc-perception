#!/usr/bin/env python3
"""Create a compact rsync-ready release bundle from validated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--integer", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--nanocockpit-package", type=Path, required=True)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--bias-audit", type=Path)
    parser.add_argument(
        "--crazyflie-build",
        type=Path,
        help="Optional successful controller build directory containing cf21bl.bin/hex.",
    )
    parser.add_argument("--l2-audit", type=Path)
    parser.add_argument("--decoder-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    integer_root = args.integer / "integer"
    sources = {
        "model/selected.pt": args.checkpoint,
        "metrics/synthetic_test.json": args.evaluation / "test_metrics.json",
        "metrics/real_flight08.json": args.evaluation / "real_flights.json",
        "metrics/integer_heldout.json": integer_root / "heldout_parity_metrics.json",
        "nanocockpit/manifest.json": args.nanocockpit_package / "manifest.json",
    }
    if args.shared:
        sources.update(
            {
                "integer/nemo_report.json": args.integer
                / "integer/nemo_stdc_shared_report.json",
                "integer/encoder_int.onnx": args.integer
                / "integer/encoder/encoder_int.onnx",
                "integer/corner_head_int.onnx": args.integer
                / "integer/corner_head/corner_head_int.onnx",
                "integer/danger_head_int.onnx": args.integer
                / "integer/danger_head/danger_head_int.onnx",
                "dory/encoder_report.json": args.export / "encoder_dory_report.json",
                "dory/corner_head_report.json": args.export
                / "corner_head_dory_report.json",
                "dory/danger_head_report.json": args.export
                / "danger_head_dory_report.json",
            }
        )
    else:
        sources.update(
            {
                "integer/nemo_report.json": args.integer
                / "integer/nemo_stdc_report.json",
                "integer/corner_int.onnx": args.integer
                / "integer/corner/corner_int.onnx",
                "integer/danger_int.onnx": args.integer
                / "integer/danger/danger_int.onnx",
                "dory/corner_report.json": args.export / "corner_dory_report.json",
                "dory/danger_report.json": args.export / "danger_dory_report.json",
            }
        )
    if args.bias_audit:
        sources["metrics/real_flight_bias.json"] = args.bias_audit
    if args.crazyflie_build:
        sources.update(
            {
                "firmware/cf21bl.bin": args.crazyflie_build / "cf21bl.bin",
                "firmware/cf21bl.hex": args.crazyflie_build / "cf21bl.hex",
            }
        )
    if args.l2_audit:
        sources["dory/l2_workspace_audit.json"] = args.l2_audit
    if args.decoder_audit:
        sources["nanocockpit/output_decoder_audit.json"] = args.decoder_audit
    optional_graphs = (
        ("encoder", "corner_head", "danger_head")
        if args.shared
        else ("corner", "danger")
    )
    optional_sources = {
        f"dory/{graph}_gvsoc_checksum.log": args.export
        / f"{graph}_gvsoc_checksum_release.log"
        for graph in optional_graphs
    }
    sources.update(
        {
            relative: source
            for relative, source in optional_sources.items()
            if source.is_file()
            and "Checksum OK" in source.read_text(errors="replace")
        }
    )
    files = []
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "source": str(source.resolve()),
            }
        )
    package_archive = shutil.make_archive(
        str(
            args.output
            / (
                "nanocockpit/gap8-stdc-real-dory-shared"
                if args.shared
                else "nanocockpit/gap8-stdc-real-dory-pair"
            )
        ),
        "gztar",
        root_dir=args.nanocockpit_package.parent,
        base_dir=args.nanocockpit_package.name,
    )
    archive = Path(package_archive)
    files.append(
        {
            "path": str(archive.relative_to(args.output)),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "source": str(args.nanocockpit_package.resolve()),
        }
    )
    manifest = {
        "format": "gap8-stdc-real-release-v2",
        "selected_model": (
            "shared_dory_frozen_real_v1" if args.shared else "real_v3"
        ),
        "selection_note": (
            "flight-06 training, flight-07 selection, untouched flight-08 "
            "test; post-selection compressed refit was evaluated and rejected"
        ),
        "files": files,
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
