"""Shared channel and decode constants for the 12x15x20 student output."""

from __future__ import annotations

from pathlib import Path

OUTPUT_CHANNELS = 12
INPUT_WIDTH, INPUT_HEIGHT = 160, 120
SENSOR_CROP_TOP = 20
HEATMAP_TL, HEATMAP_TR, HEATMAP_BR, HEATMAP_BL = range(4)
OFFSET_BASE = 4
CONFIDENCE_BASE = 8
GRID_WIDTH, GRID_HEIGHT = 20, 15
SCORE_LIMIT = 6.0
TERMINAL_SCORE_OFFSET = SCORE_LIMIT
OFFSET_MIN, OFFSET_MAX = 0.0, 6.0
# The HM01B0 160x120 crop is approximately [-42.3, +41.2] degrees after
# calibration.  Keep every fixed-normal ray inside its measured support.
NORMAL_ANGLES_DEG = (-40.0, -13.333333, 13.333333, 40.0)


def decode_scalar_fields(output):
    """Decode spatially averaged offsets/confidences from an NCHW tensor."""
    if tuple(output.shape[-3:]) != (OUTPUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"expected (*,12,15,20), got {tuple(output.shape)}")
    offset_scores = output[:, OFFSET_BASE:OFFSET_BASE + 4].mean(dim=(-2, -1))
    confidence_scores = output[:, CONFIDENCE_BASE:CONFIDENCE_BASE + 4].mean(dim=(-2, -1))
    clipped = offset_scores.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    offsets = OFFSET_MIN + (clipped + SCORE_LIMIT) / (2 * SCORE_LIMIT) * (OFFSET_MAX - OFFSET_MIN)
    return offsets, confidence_scores


def write_c_header(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""/* Generated from gap8_perception.output_contract. */
#ifndef GAP8_PERCEPTION_OUTPUT_CONTRACT_H
#define GAP8_PERCEPTION_OUTPUT_CONTRACT_H
#define PERCEPTION_INPUT_WIDTH 160
#define PERCEPTION_INPUT_HEIGHT 120
#define PERCEPTION_SENSOR_CROP_TOP 20
#define PERCEPTION_OUTPUT_CHANNELS 12
#define PERCEPTION_GRID_WIDTH 20
#define PERCEPTION_GRID_HEIGHT 15
#define PERCEPTION_HEATMAP_TL 0
#define PERCEPTION_HEATMAP_TR 1
#define PERCEPTION_HEATMAP_BR 2
#define PERCEPTION_HEATMAP_BL 3
#define PERCEPTION_OFFSET_BASE 4
#define PERCEPTION_CONFIDENCE_BASE 8
#define PERCEPTION_SCORE_LIMIT 6.0f
#define PERCEPTION_CONFIDENCE_THRESHOLD 0.0f
#define PERCEPTION_TERMINAL_SCORE_OFFSET 6.0f
#define PERCEPTION_OFFSET_MIN 0.0f
#define PERCEPTION_OFFSET_MAX 6.0f
#define PERCEPTION_NORMAL_0_DEG -40.0f
#define PERCEPTION_NORMAL_1_DEG -13.333333f
#define PERCEPTION_NORMAL_2_DEG 13.333333f
#define PERCEPTION_NORMAL_3_DEG 40.0f
#endif
""")
