# Proposed STDC-FPN perception network

## Purpose

`ProposedSTDCFPNNet` is the 160 x 120 x 1 candidate introduced for the next
GAP8/NanoCockpit perception release.  It preserves fine gate-corner detail
while giving dense danger prediction access to the 10 x 8 receptive field.

## Tensor contract

Shapes below use `C x H x W`.

| Stage | Operator | Output |
|---|---|---|
| input | HM01B0 center crop | 1 x 120 x 160 |
| stem | 3x3 stride-2 Conv, BN, ReLU | 16 x 60 x 80 |
| E1 | modified strided STDC | 32 x 30 x 40 |
| E2 | modified strided STDC | 64 x 15 x 20 |
| E3 | modified strided STDC | 96 x 8 x 10 |
| P3/P2 | 1x1 projections, resize, concatenate, 3x3 depthwise-separable fuse | 16 x 15 x 20 |
| P1/F40 | resize, 1x1 projection, elementwise add, 3x3 depthwise-separable fuse | 16 x 30 x 40 |
| corner head | 1x1 Conv | 4 x 30 x 40 |
| danger head | 1x1 Conv | 1 x 30 x 40 |

Corner channels are always `TL, TR, BR, BL`.  Coordinates are decoded on the
160 x 120 crop, then shifted by 20 sensor rows when reported in the original
160 x 160 HM01B0 image.

## Training-only supervision

The deployed forward path emits only corners and danger.  While training, a
depthwise-3x3 plus pointwise-1x1 boundary head supervises the gate-opening
boundary and an optional FC head regresses eight normalized corner
coordinates from the E3 projection.  Both heads are excluded from export.

Synthetic data supplies corner heatmaps and conservative collision targets.
The dense danger target is 40 x 30; for comparison with the existing DORY
release it is adaptively max-reduced to 10 x 8.  This preserves a hazardous
source cell.  Real flights have gate-corner labels but no obstacle/collision
labels, so mixed adaptation uses them for corners only and retains synthetic
danger supervision.

## Real-flight protocol

Use flight 06 for real adaptation, flight 07 for selection, and flight 08
only once for final reporting.  This is a whole-flight split, not a random
frame split, to avoid temporal leakage.  Frames outside the 20:140 crop are
excluded rather than relabelled.

## Deployment partition

The full FPN uses `Resize` and feature fusion, which stock DORY cannot accept
as one graph.  Deployment therefore partitions at pyramid boundaries.  DORY
tiles the convolutional stages; the NanoCockpit glue owns nearest-neighbour
copies, explicit channel packing/unpacking, and the E1 skip addition.  Each
partition is INT8-quantized with NeMO and must pass float-to-integer parity,
DORY frontend validation, generated-tile L1 audit, GVSOC checksum, and the
NanoCockpit package test before release.

The existing shared-DORY release remains the fallback until these gates pass.

## Candidate v1 result and release decision

The completed v1 training artifacts are stored outside the source repository
under `workspace/gap8_proposed_stdc_fpn_v1` and
`workspace/gap8_proposed_stdc_fpn_real_v1`.  Against the selected shared-DORY
baseline on the same 5,000-image synthetic test set, pretraining improved mean
corner error by 3.16 px (3.16 px absolute) and PCK@4 by 0.0401.  It regressed
the comparable conservative 10x8 danger recall by 0.0352 and IoU by 0.1337.

After mixed real adaptation (flight 06 train, flight 07 select), synthetic
corner error remained better than the baseline by 1.85 px, but flight 08
corner error was 22.39 px mean over 3,045 in-crop labeled frames.  The selected
shared-DORY release is 12.79 px mean on that test.  The candidate also lacks a
completed NeMO/DORY integer-parity, tiling, GVSOC, and NanoCockpit package
chain.  It is therefore **rejected for deployment release v1**.  The checked-in
source, run recipes, logs, and reports are retained so a future revision can
address danger calibration and real-flight domain transfer without losing this
negative result.
