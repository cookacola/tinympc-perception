"""Label-preserving approximation of the measured HM01B0 flight image domain.

This is deliberately applied only to the mono training image.  Geometry labels
remain tied to the renderer; any augmentation which destroys observability
returns an explicit invalid-observation flag for the offset/confidence loss.
"""

from __future__ import annotations

import cv2
import numpy as np


def _motion_kernel(length: int, angle_rad: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    center = (length - 1) * 0.5
    dx, dy = np.cos(angle_rad) * center, np.sin(angle_rad) * center
    cv2.line(kernel, (int(round(center - dx)), int(round(center - dy))),
             (int(round(center + dx)), int(round(center + dy))), 1.0, 1)
    return kernel / kernel.sum().clip(min=1.0)


def augment_hm01b0(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Return an HM01B0-like frame and whether safety observation remains valid.

    The renderer owns episode-level exposure and room lighting.  This function
    supplies only bounded capture noise and occasional motion blur. The guard
    geometry is rendered once by the calibrated camera rig. ``False`` means the caller must not
    supervise free-range or confidence values from a deliberately unusable
    capture.
    """
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("expected uint8 grayscale image")
    if rng.random() > probability:
        return image, True
    value = image.astype(np.float32) / 255.0
    height, width = value.shape
    # Keep per-frame variability modest. Wide, independently sampled exposure
    # made adjacent synthetic images implausibly different and caused the
    # saturated examples the review caught.
    gamma = float(rng.uniform(0.88, 1.10))
    gain = float(rng.uniform(0.88, 1.18))
    black = float(rng.uniform(-0.020, 0.035))
    value = gain * np.power(np.clip(value, 0.0, 1.0), gamma) + black
    # Lens vignetting plus low-frequency per-frame illumination variation.
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    radius = np.sqrt(((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
    value *= 1.0 - float(rng.uniform(0.03, 0.15)) * radius ** 2
    # Fixed-pattern noise is structured rather than i.i.d. Gaussian noise.
    value += rng.normal(0.0, rng.uniform(0.0, 0.009), (height, 1))
    value += rng.normal(0.0, rng.uniform(0.0, 0.006), (1, width))
    # Shot noise rises with intensity; read noise remains in dark regions.
    value += rng.normal(0.0, rng.uniform(0.006, 0.016), value.shape) * np.sqrt(np.clip(value, 0.0, 1.0) + 0.04)
    if rng.random() < 0.20:
        value = cv2.filter2D(value, -1, _motion_kernel(int(rng.choice((3, 5, 7))), float(rng.uniform(0, np.pi))))
    invalid = rng.random() < 0.02
    if invalid:
        # Saturated/underexposed/obscured frames represent "not observable",
        # not an obstacle at zero distance.
        if rng.random() < 0.5:
            value[:] = rng.uniform(0.88, 1.0)
        else:
            value[:] = rng.uniform(0.0, 0.08)
    return np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8), not invalid
