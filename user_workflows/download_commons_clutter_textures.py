#!/usr/bin/env python3
"""Cache license-audited clutter photographs from Wikimedia Commons."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "TinyMPCResearchDataset/1.0 "
    "(https://github.com/cookacola/tinympc-perception; synthetic texture cache)"
)
ALLOWED_LICENSES = {
    "CC0",
    "Public domain",
    "CC BY 1.0",
    "CC BY 2.0",
    "CC BY 2.5",
    "CC BY 3.0",
    "CC BY 4.0",
}
QUERY_GROUPS = {
    "train": (
        "cluttered workshop interior",
        "messy garage interior",
        "warehouse shelves interior",
        "storage room interior",
        "office desk clutter",
        "classroom interior empty",
        "industrial pipes wall",
        "tools hanging on wall",
    ),
    "validation": (
        "utility room interior",
        "laboratory shelves interior",
    ),
    "test": (
        "hobby room interior",
        "laundry room shelves interior",
    ),
}


def clean_markup(value: str | None) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())


def open_with_backoff(request: urllib.request.Request, timeout: int) -> bytes:
    delays = (0.0, 2.0, 5.0, 10.0)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == len(delays):
                raise
    raise RuntimeError("unreachable retry state")


def api_query(parameters: dict[str, str | int]) -> dict:
    encoded = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    request = urllib.request.Request(f"{API}?{encoded}", headers={"User-Agent": USER_AGENT})
    return json.loads(open_with_backoff(request, 45))


def search(query: str, limit: int) -> list[dict]:
    payload = api_query({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(30, max(limit * 3, 10)),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 640,
    })
    return payload.get("query", {}).get("pages", [])


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return open_with_backoff(request, 60)


def prepare_image(raw: bytes, maximum_side: int) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 256:
        return None
    scale = min(1.0, maximum_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def metadata_value(metadata: dict, name: str) -> str:
    entry = metadata.get(name, {})
    return clean_markup(entry.get("value", "") if isinstance(entry, dict) else str(entry))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument("--maximum-side", type=int, default=1024)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.per_query < 1 or arguments.maximum_side < 256:
        raise ValueError("per-query must be positive and maximum-side must be at least 256")
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
            accepted = sum(1 for asset in assets if asset["split"] == split and asset["query"] == query)
            try:
                pages = search(query, arguments.per_query)
            except urllib.error.HTTPError as error:
                print(json.dumps({"query": query, "api_error": str(error)}), flush=True)
                break
            for page in pages:
                if accepted >= arguments.per_query:
                    break
                info_items = page.get("imageinfo", [])
                if not info_items:
                    continue
                info = info_items[0]
                metadata = info.get("extmetadata", {})
                license_name = metadata_value(metadata, "LicenseShortName")
                if license_name not in ALLOWED_LICENSES:
                    continue
                if info.get("mime") not in {"image/jpeg", "image/png"}:
                    continue
                commons_page = info.get("descriptionurl") or info.get("descriptionshorturl")
                image_url = info.get("thumburl") or info.get("url")
                if not commons_page or not image_url or commons_page in known_pages:
                    continue
                try:
                    time.sleep(arguments.delay_seconds)
                    raw = download(image_url)
                    image = prepare_image(raw, arguments.maximum_side)
                except urllib.error.HTTPError as error:
                    print(json.dumps({"query": query, "download_error": str(error), "url": image_url}), flush=True)
                    if error.code == 429:
                        break
                    continue
                except (OSError, ValueError) as error:
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
                    "title": page.get("title", ""),
                    "commons_page": commons_page,
                    "download_url": image_url,
                    "license": license_name,
                    "license_url": metadata_value(metadata, "LicenseUrl"),
                    "artist": metadata_value(metadata, "Artist"),
                    "credit": metadata_value(metadata, "Credit"),
                    "attribution": metadata_value(metadata, "Attribution"),
                    "source_sha256": source_hash,
                    "prepared_resolution": [int(image.shape[1]), int(image.shape[0])],
                    "modification": "resized if necessary and JPEG re-encoded; used as a synthetic-scene material",
                }
                assets.append(asset)
                known_pages.add(commons_page)
                known_hashes.add(source_hash)
                accepted += 1
                manifest_path.write_text(json.dumps({
                    "source": "Wikimedia Commons API",
                    "source_policy": "CC0, public domain, or CC BY only; CC BY-SA excluded",
                    "queries_by_split": QUERY_GROUPS,
                    "assets": assets,
                }, indent=2) + "\n")
                print(json.dumps({"split": split, "query": query, "accepted": filename}), flush=True)

    counts = {split: sum(asset["split"] == split for asset in assets) for split in QUERY_GROUPS}
    print(json.dumps({"manifest": str(manifest_path), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
