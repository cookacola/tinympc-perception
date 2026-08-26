# DORY motion-conditioned TTC and gate release v1

This release pins the selected critical-TTC-first float checkpoint trained on the
100k Isaac Sim kinematic corpus. It uses a DORY-compatible, three-graph network
and retains the shared gate head. It is the recommended checkpoint for the next
TTC experiment, but it is **not yet a quantized flight release**.

## Selected checkpoint

- `model/ttc_motion_gate_dory_v1.pt`
- SHA-256: `24c36229bdfedb755726c811d2812bf0b10cf337c32cd1c2718c2ffbec302adc`
- selected upstream training epoch: 13
- continuation selection epoch: 0 (the warm-start remained best during the
  five-epoch GPU continuation)
- dataset split: 74,106 train / 16,829 validation / 14,040 test examples

The image input is two consecutive 160 x 160 grayscale frames in
`[previous,current]` order. The ten onboard inputs are body velocity (3), angular
velocity (3), gravity direction in body coordinates (3), and frame interval
(1). Thus angular velocity and an attitude proxy are explicitly supplied.

The learned deployment is partitioned into three static graphs:

1. encoder: `2 x 160 x 160 -> 64 x 20 x 20`
2. gate head: `64 x 20 x 20 -> 8 x 20 x 20`
3. TTC head: packed `74 x 20 x 20 -> 7 x 20 x 20`

The TTC input packs the 64 encoder channels with ten normalized state planes.
Host code performs packing and output decoding; those operations are not part of
the learned graphs.

## Float held-out results

The critical definition is valid, approaching motion with inverse TTC at least
2.0 s^-1, equivalent to TTC at most 0.5 s.

- general inverse-TTC MAE: 0.17602 s^-1
- approaching inverse-TTC MAE: 0.19786 s^-1
- critical-region inverse-TTC MAE: 1.12939 s^-1
- regression-derived critical precision / recall: 0.53366 / 0.77928
- risk-head precision / recall at threshold 0.552: 0.60774 / 0.72639
- risk-head precision / recall at the validation-frozen 0.57237 threshold:
  0.62561 / 0.70559
- gate PCK@8 over supervised visible corners: 0.79631
- full-gate PCK@8: 0.87652
- gate visibility F1: 0.84896

Relative to the prior rich-graph critical-first candidate, this backend-safe
model improved test critical inverse-TTC MAE by 1.43%, regression critical
recall by 1.75 percentage points, and risk-head recall at 0.552 by 1.08 points.
Its general inverse-TTC MAE is 0.00134 s^-1 worse. It did not satisfy the strict
general-metric retention guard, so this is explicitly a critical-first tradeoff.

`metrics/critical_threshold_calibration.json` contains only thresholds selected
on validation and then frozen for test. Threshold 0.57237 is the balanced
operating point. Threshold 0.552 gives more recall; 0.51861 gives still more
recall (0.75844 on test) at lower precision (0.57756).

## Gate supervision policy

Gate loss is masked for far or subresolution gates: selected gates must have at
least one visible corner, be no farther than 8 m, span at least 16 pixels in
both projected dimensions, and have at least 256 px^2 projected area. These
frames still train TTC and are not relabeled as no-gate negatives. Close partial
gates with one, two, or three visible corners remain supervised, as do genuine
no-gate negative frames.

## DORY acceptance and limitation

The float ONNX audit passed using only Conv, ReLU, Add, Identity, and Constant.
All three graphs then passed NeMO int8 export, the DORY GAP8 frontend and tiler,
generated-C compilation, and layer-by-layer GAP8 GVSOC checksums:

- encoder: 22/22 checksums, 16.192 M MACs, 12,320,727 cycles
- gate head: 4/4 checksums, 1.1904 M MACs, 554,380 cycles
- TTC head: 13/13 checksums, 8.640 M MACs, 3,912,586 cycles

This proves compiler and simulator acceptance, not physical GAP8 or flight
approval. Post-training int8 quantization still changes the outputs materially:
inverse-TTC softplus MAE versus float is 0.23693 s^-1, risk-class agreement is
0.85177, and gate heatmap peak agreement is poor (15.625% exact; 4.69-cell mean
displacement). The int8 artifact should therefore be treated as a deployment
pipeline acceptance artifact. TTC/risk-focused quantization-aware training is
the next recommended experiment before hardware deployment.

The montages under `artifacts/` intentionally include true detections, a
critical miss, a safe example, a false alarm, and partial-gate cases.
