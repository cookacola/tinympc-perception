# Historical HM01B0 v4 pipeline (non-canonical, 2026-07-27)

This report concerns a superseded dense-danger model. The canonical contract
is the sequential 12-channel completed CNN design.

The selected float model is the retained-gate `Gap8PackedMultiTaskNet` with
7,784 parameters and 11.904 M MACs. Its input is the native 160×160
monochrome HM01B0 frame. Outputs are four ordered 40×40 corner heatmaps plus
40×40 packed channels that conservatively reduce to 20×20 nominal collision
probability, inverse range, uncertainty, and gate-opening probability.

Authoritative artifact roots:

- corrected render:
  `workspace/course_hm01b0_nbd_hm01b0_v2_50k`;
- exact targets:
  `workspace/gap8_rollout_targets_hm01b0_v4`;
- audit:
  `workspace/gap8_rollout_audit_hm01b0_v4`;
- float:
  `workspace/gap8_rollout_float_hm01b0_v4`;
- PyTorch QAT:
  `workspace/gap8_rollout_qat_hm01b0_v4`;
- no-gate ablation:
  `workspace/gap8_rollout_float_no_gate_hm01b0_v4`;
- selected NeMO QAT checkpoint:
  `workspace/gap8_nemo_distill24_hm01b0_v4/epoch_22_nemo_qat.pt`;
- deterministic NeMO/DORY package:
  `workspace/gap8_rollout_nemo_dory_hm01b0_v4_epoch22_seed9`;
- compact handoff bundle:
  `workspace/gap8_download_bundle_hm01b0_v4`.

## Measured status

The corrected dataset contains 50,000 unique frames in 50 complete shards.
All records and exact rollout targets pass the full audit. The render uses the
supplied HM01B0 calibration and the textured NewBeeDrone gates. The train
corner-valid fraction is 76.76%. Nominal 1 m/s collision-positive cells are
51.09%; 0.5 m/s and 5 m/s stress variants are 29.96% and 98.52%.

The selected float model's held-out test results are:

- mean corner error: 2.995 image pixels;
- PCK@4: 0.887;
- gate IoU: 0.882;
- nominal danger precision/recall/IoU: 0.827/0.944/0.788;
- nominal false-safe rate: 0.056;
- urgency MAE: 0.146.

The no-gate ablation improves danger IoU only from 0.788 to 0.793 while
degrading mean corners to 4.66 px and false gate detections to 0.665. The gate
head is therefore retained. PyTorch QAT is also inferior to the float model
for danger safety and is not the deployed checkpoint.

NeMO-native distillation epoch 22 is the selected integer checkpoint.
Old NeMO traverses parts of its graph in hash order, so the validated export
uses `PYTHONHASHSEED=9`. Two independent exports under that seed produce the
same ONNX SHA-256. Across 32 fixed parity images, the integer model has:

- 3.471 heatmap-pixel mean corner-peak displacement from float;
- 0.1601 mean absolute danger-probability error;
- 0.00413 gate-probability MAE.

The production export rejects candidates above 4.5 heatmap pixels or 0.18
danger-probability MAE. DORY generated 48 hardware nodes, 11.904 M MACs, and a
36,240-byte maximum L1 tile estimate against 64 KiB capacity.

NanoCockpit firmware and the `vision`-branch Crazyflie/TinyMPC firmware both
compile with the selected integer affine constants. The controller-side map
protocol is version 2: a 222-byte CRC-protected packet carries uint4 10×10
collision, range, uncertainty, and gate-opening maps. Gate opening is a
permission signal and is min-pooled, while hazards are max-pooled.

The controller only reduces danger inside a synchronized, inset gate polygon
when gate-opening confidence is high, uncertainty is low, and predicted free
range extends beyond the known gate plane. An obstacle inside the opening
therefore remains dangerous. Image constraints are active by default:

```text
percept.enable=1
percept.constrain=1
percept.logOnly=0
percept.gateOpen=1
```

All 27 perception tests and 14 controller integration tests pass.

## Unresolved deployment evidence

The cycle-accurate GVSOC checksum job is still authoritative only after it
prints `Checking final layer: Checksum OK`; the current run is Slurm job 2683.
Physical GAP8/Crazyflie flashing, measured camera-to-control latency, and
closed-loop controller ablations A–F require hardware or a controller-in-loop
simulator and are not established by compilation.

The gate-only real-flight verification contains no obstacles and cannot
validate danger. On 12,630 labeled real frames, the float model has about
41.2 px mean raw corner error. Labels are imperfect, but the prediction
montage confirms a substantial sim-to-real gap, so real-flight gate
localization remains unsuitable without mixed synthetic/real adaptation and
fresh verification.
