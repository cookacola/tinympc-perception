#!/usr/bin/env python3
"""Cache openly licensed clutter images returned by the Openverse API."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

from download_commons_clutter_textures import QUERY_GROUPS, open_with_backoff, prepare_image


API = "https://api.openverse.org/v1/images/"
USER_AGENT = (
    "TinyMPCResearchDataset/1.0 "
    "(https://github.com/cookacola/tinympc-perception; synthetic texture cache)"
)
LICENSE_NAMES = {"cc0": "CC0", "pdm": "Public domain", "by": "CC BY"}


def search(query: str, page_size: int) -> list[dict]:
    parameters = urllib.parse.urlencode({
        "q": query,
        "license": "cc0,pdm,by",
        "extension": "jpg,png,jpeg",
        "category": "photograph",
        "mature": "false",
        "page_size": page_size,
    })
    request = urllib.request.Request(f"{API}?{parameters}", headers={"User-Agent": USER_AGENT})
    return json.loads(open_with_backoff(request, 45)).get("results", [])


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return open_with_backoff(request, 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument("--maximum-side", type=int, default=1024)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"assets": []}
    assets = list(existing.get("assets", []))
    known_pages = {asset["commons_page"] for asset in assets}
    known_hashes = {asset["source_sha256"] for asset in assets}

    for split, queries in QUERY_GROUPS.items():
        split_dir = arguments.output / split
        split_dir.mkdir(exist_ok=True)
        for query in queries:
            accepted = sum(
                asset["split"] == split and asset["query"] == query and asset.get("catalog") == "Openverse"
                for asset in assets
            )
            for result in search(query, max(10, arguments.per_query * 4)):
                if accepted >= arguments.per_query:
                    break
                license_code = str(result.get("license", "")).lower()
                if license_code not in LICENSE_NAMES:
                    continue
                source_page = result.get("foreign_landing_url") or result.get("detail_url")
                image_url = result.get("thumbnail") or result.get("url")
                if not source_page or not image_url or source_page in known_pages:
                    continue
                try:
                    time.sleep(arguments.delay_seconds)
                    raw = download(image_url)
                    image = prepare_image(raw, arguments.maximum_side)
                except OSError as error:
                    print(json.dumps({"query": query, "skipped": str(error), "url": image_url}), flush=True)
                    continue
                if image is None:
                    continue
                source_hash = hashlib.sha256(raw).hexdigest()
                if source_hash in known_hashes:
                    continue
                filename = f"{source_hash[:16]}.jpg"
                destination = split_dir / filename
                if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"failed to write {destination}")
                asset = {
                    "file": str(destination.relative_to(arguments.output)),
                    "split": split,
                    "query": query,
                    "catalog": "Openverse",
                    "source": result.get("source"),
                    "title": result.get("title") or "",
                    "commons_page": source_page,
                    "download_url": image_url,
                    "license": LICENSE_NAMES[license_code] + (
                        f" {result.get('license_version')}" if result.get("license_version") else ""
                    ),
                    "license_url": result.get("license_url") or "",
                    "artist": result.get("creator") or "",
                    "credit": result.get("attribution") or "",
                    "attribution": result.get("attribution") or "",
                    "source_sha256": source_hash,
                    "prepared_resolution": [int(image.shape[1]), int(image.shape[0])],
                    "modification": "resized if necessary and JPEG re-encoded; used as a synthetic-scene material",
                }
                assets.append(asset)
                known_pages.add(source_page)
                known_hashes.add(source_hash)
                accepted += 1
                manifest_path.write_text(json.dumps({
                    "sources": ["Wikimedia Commons API", "Openverse API"],
                    "source_policy": "CC0, public domain, or CC BY only; attribution retained",
                    "queries_by_split": QUERY_GROUPS,
                    "assets": assets,
                }, indent=2) + "\n")
                print(json.dumps({"split": split, "query": query, "accepted": filename}), flush=True)

    counts = {split: sum(asset["split"] == split for asset in assets) for split in QUERY_GROUPS}
    print(json.dumps({"manifest": str(manifest_path), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
