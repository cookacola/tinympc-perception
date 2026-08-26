"""Targets, losses, and coordinate decoding for the four-corner gate branch."""
from __future__ import annotations

import torch
from torch.nn import functional as F


OUTPUT_SIZE = 20
IMAGE_SIZE = 160
CELL_SIZE = IMAGE_SIZE / OUTPUT_SIZE
CORNER_ORDER = ("TL", "TR", "BR", "BL")


def output_pixel_centers(reference):
    centers = torch.arange(OUTPUT_SIZE, device=reference.device, dtype=reference.dtype)
    centers = centers * CELL_SIZE + (CELL_SIZE - 1.0) / 2.0
    return torch.meshgrid(centers, centers, indexing="ij")


def gate_heatmap_targets(corners_px, visible, sigma_px=8.0):
    """Build four Gaussian 20x20 targets at TL,TR,BR,BL rail-center landmarks."""
    grid_y, grid_x = output_pixel_centers(corners_px)
    dx = grid_x[None, None] - corners_px[:, :, 0, None, None]
    dy = grid_y[None, None] - corners_px[:, :, 1, None, None]
    heatmaps = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma_px ** 2))
    return heatmaps * visible[:, :, None, None].to(heatmaps.dtype)


def softargmax_gate_coordinates(heatmap_logits):
    batch, corners, height, width = heatmap_logits.shape
    if (height, width) != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(f"expected 20x20 heatmaps, received {(height, width)}")
    probability = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(heatmap_logits)
    grid_y, grid_x = output_pixel_centers(heatmap_logits)
    x = (probability * grid_x).sum((-2, -1))
    y = (probability * grid_y).sum((-2, -1))
    return torch.stack((x, y), dim=-1)


def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gate_perception_loss(
    prediction,
    target,
    heatmap_weight=1.0,
    coordinate_weight=0.5,
    visibility_weight=0.25,
):
    visible = target["gate_corners_visible"].bool()
    corners = target["gate_corners_px"]
    logits = prediction["gate_heatmap_logits"]
    heatmaps = gate_heatmap_targets(corners, visible)
    pixel_weight = 1.0 + 10.0 * heatmaps
    heatmap_values = F.binary_cross_entropy_with_logits(logits, heatmaps, reduction="none")
    heatmap_mask = visible[:, :, None, None].expand_as(heatmap_values)
    heatmap = _masked_mean(heatmap_values * pixel_weight, heatmap_mask)
    decoded = softargmax_gate_coordinates(logits)
    coordinate_values = F.smooth_l1_loss(
        decoded / CELL_SIZE, corners / CELL_SIZE, reduction="none"
    ).mean(-1)
    coordinate = _masked_mean(coordinate_values, visible)
    visibility = F.binary_cross_entropy_with_logits(
        prediction["gate_visibility_logits"], visible.to(logits.dtype)
    )
    total = (
        heatmap_weight * heatmap
        + coordinate_weight * coordinate
        + visibility_weight * visibility
    )
    return total, {
        "heatmap": heatmap,
        "coordinate": coordinate,
        "visibility": visibility,
    }
