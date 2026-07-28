#!/usr/bin/env python3
"""Create renderer-safe HM01B0-oriented gate textures from known-good PNGs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "assets/gates"
OUTPUT = ROOT / "newbeedrone_hm01b0_v2"


def hm01b0_luminance(image: np.ndarray) -> np.ndarray:
    """Lift dark fabric while preserving printed texture and exact branding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The real HM01B0/NIR capture resolves black fabric as dark gray rather
    # than zero. This monotonic camera-response approximation makes seams,
    # weave, orange panels, and logos survive a final 160x160 render.
    source = np.array([0, 24, 55, 100, 160, 220, 255], np.float32)
    destination = np.array([28, 55, 88, 135, 182, 225, 250], np.float32)
    lifted = np.interp(gray, source, destination).astype(np.uint8)
    local = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(16, 4)).apply(lifted)
    mixed = np.uint8(
        np.clip(0.72 * lifted.astype(np.float32) + 0.28 * local, 0, 255)
    )
    return cv2.cvtColor(mixed, cv2.COLOR_GRAY2BGR)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for part in ("top", "bottom", "left", "right"):
        source = ROOT / f"newbeedrone_{part}_v1.png"
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        destination = OUTPUT / f"newbeedrone_{part}_v2.png"
        if not cv2.imwrite(
            str(destination), hm01b0_luminance(image),
            [cv2.IMWRITE_PNG_COMPRESSION, 6],
        ):
            raise RuntimeError(f"failed to write {destination}")
        check = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if check is None or check.shape != image.shape:
            raise RuntimeError(f"invalid output texture: {destination}")
        print(f"{part}: {image.shape} -> {destination}")

    atlas_source = ROOT / "newbeedrone_square_atlas_v1.png"
    atlas = cv2.imread(str(atlas_source), cv2.IMREAD_COLOR)
    atlas_output = OUTPUT / "newbeedrone_square_atlas_v2.png"
    if atlas is None or not cv2.imwrite(
        str(atlas_output), hm01b0_luminance(atlas),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    ):
        raise RuntimeError("failed to create v2 atlas")


if __name__ == "__main__":
    main()
