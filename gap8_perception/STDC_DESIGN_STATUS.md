# 160x120 STDC multi-head network

This is the new network implementation corresponding to the supplied
Multi-Headed CNN Design Document. It is independent of the earlier
`Gap8MultiTaskNet` deployment.

## Implemented contract

- Input: one 160x120 monochrome image, represented as NCHW `[1,1,120,160]`.
- Stem: stride-two 3x3 convolution followed by depthwise/pointwise convolution.
- Shared encoder: 32-, 64-, and 96-channel stride-two STDC-inspired stages.
- Corner head: stage-3 + stage-2 fusion, followed by stage-1 fusion, producing
  ordered TL/TR/BR/BL 40x30 heatmaps.
- Danger head: stage-3 + stage-2 fusion producing a graded 20x15 map.
- Corner confidence: the calibrated peak of each corresponding heatmap.
- Loss: focal weighted heatmap MSE plus positive-weighted BCE, Soft Dice, and
  continuous-risk regression for danger.
- Phantom-gate rejection: confidence, semantic corner order, convexity,
  minimum area, and side-ratio gates.
- Gate/danger arbitration: only an accepted inset quadrilateral can reduce
  danger; rejected geometry is a strict no-op.
- Privileged teacher: a training-only second input contains normalized inverse
  metric depth. A mono student is fine-tuned from the teacher with supervised
  and output-distillation losses.
- INT8 preparation: PyTorch QAT is run after the selected float checkpoint.

## Static resource evidence

`python -m gap8_perception.profile_stdc` reports:

| Quantity | Measured | Design constraint |
|---|---:|---:|
| Parameters | 117,621 | about 100k-180k |
| Convolution MACs | 23,577,600 | below about 20m-30m |
| Largest untiled INT8 tensor | 76,800 B | 64 kB L1 |

The untiled stem tensor exceeds L1. This is not a deployment pass or failure:
DORY must tile it, and the generated tiling report must prove that every live
L1 tile fits. We do not substitute the single-tensor estimate for that test.

## Data scope

The authoritative existing synthetic corpus contains 50,000 corrected,
unique HM01B0 frames and exact rollout targets. It uses the supplied camera
calibration and NewBeeDrone textures, but one fixed lab course rather than
750,000 images across varying racetracks.

`user_workflows/run_varied_course_dataset.slurm` now provides the missing
varying-track generator path. Its deterministic per-shard layouts randomize
gate position/height, obstacle count/geometry/location, lighting, and floor
appearance while retaining aligned HM01B0 RGB, depth, and semantic outputs.
Two GPU smoke shards passed alignment validation. The full 750,000-frame
render has not been claimed: measured storage is about 70 KiB/frame, while
the current workspace has only about 20 GiB free versus roughly 64 GiB
required including a safety reserve. The launcher refuses to partially fill
the filesystem and can be pointed at a larger `OUTPUT_ROOT`.

The real-flight set has 12,630 labeled gate frames. It contains no obstacles,
so it can evaluate corner domain transfer but cannot train or validate danger
without inventing negative labels. Frames whose labeled gate leaves the
160x120 center crop are reported as excluded.

## Training and acceptance sequence

1. Train an augmented mono float baseline.
2. Train the inverse-depth privileged teacher.
3. Distill the teacher into the mono student and compare held-out synthetic
   and real-gate metrics against the direct baseline.
4. Select on safety-first danger recall/false-negative rate, then corner error
   and phantom-gate precision.
5. Run INT8 quantization and label-aware float/integer parity on held-out data.
6. Export through NeMO and run the real DORY frontend and GAP8 tiler.
7. Compile and checksum in GVSOC, package into NanoCockpit, and compile the
   Crazyflie/TinyMPC receiver.

The design graph exports ONNX `Resize` for its top-down feature fusion. The
installed stock DORY NEMO frontend does not list `Resize` as an accepted
operator. Therefore PyTorch/QAT success is not called deployability: the graph
must either pass an implemented frontend/kernel extension or be transformed
to a numerically validated DORY-supported equivalent before completion.

The first implemented fallback was a distilled, resize-free DORY pair. The corner
graph retains the 40x30 stage-1 detail path. The danger graph retains the full
stage-3 receptive field and emits a conservative 10x8 map; its supervision is
max-reduced from 20x15 so a hazardous source cell cannot disappear through
averaging. Together the two sequential graphs measure 107,205 parameters and
25,226,880 MACs. They contain only Conv/ReLU/Add and therefore fit the proven
frontend operator surface. These are combined totals; reporting either graph
alone would understate deployment cost.

The selected deployment now removes that duplication with a shared
32x30x40 encoder feeding separate corner and danger heads. It has 102,933
parameters and 18,679,680 MACs, and its split graphs are numerically identical
to the combined PyTorch graph. The corner path is initialized from and frozen
to the validated real-adapted pair while the downstream danger stages are
trained. This preserves the selected float real-flight metrics while reducing
the deployed compute. All three graphs use only Conv/ReLU/Add.

The selected pair is `gap8_stdc_dory_pair_real_v3/selected.pt`. Its held-out
float danger recall is 0.9864 at threshold 0.5. On 512 deterministic images
from untouched synthetic test shards, naive INT8 thresholding at 0.5 reduced
recall to 0.9625. This is not accepted silently: the deployed integer threshold
is calibrated on those held-out labels to 0.110427, yielding recall 0.9916,
precision 0.8668, and false-negative rate 0.00837. This reflects the design's
explicit preference for obstacle false positives over false negatives.
Integer gate precision is 0.9818 at its confidence threshold.

DORY generated all three shared graphs successfully. DORY's maximum estimated live L1 tile
is 36,289 bytes. The namespaced NanoCockpit pair builds into one GAP8 firmware
image and uses 225,676 bytes of 512 kB static L2, including its bounded
180,000-byte directional workspace. A generated-allocation audit proves a
153,984-byte peak (153,985 bytes required by DORY's strict allocator), leaving
26,015 bytes of workspace margin. It center-crops the camera to 160x120, runs the encoder
once, rejects phantom gate geometry, and only emits an inset
gate-opening permission map after confidence and geometry acceptance.

The TinyMPC receiver consumes the calibrated conservative danger mask directly
and increases risk for speed and stale-frame reach. Its host equivalence suite
passes 14/14 tests. The repository's Bitcraze builder container also completes
the full `cf21bl` STM32 cross-build with the shared-model threshold: flash
353,492/1,032,192 bytes, RAM 116,212/131,072 bytes, and CCM
62,248/65,536 bytes. Remaining mandatory gates are the exact GVSOC
checksum/latency run and physical track validation.

## Measured model results

- Simulation-only model on real flights: about 40.7 px mean corner error.
- Rich real-adapted model, untouched flight 08: 11.45 px mean, 6.16 px median.
- Deployable DORY pair, untouched flight 08: 12.79 px mean, 6.26 px median,
  0.927 gate detection rate.
- Deployable float pair, synthetic test: 6.32 px mean corner error, 0.961 gate
  precision, 0.986 danger recall.
- Deployable INT8 pair, 512-image held-out safety audit: 0.9916 danger recall
  at the deployment threshold and 0.9818 gate precision.
- Shared deployable INT8 model: 6.44 px synthetic corner error, 0.9871 gate
  precision, and 0.9923 danger recall at its calibrated 0.07227 threshold.

A post-selection rich-model refit on flights 06+07 improved untouched flight
08 to 11.24 px mean error and 0.945 detection. Repeating that refit for the
compressed DORY corner graph increased detection to 0.989 but degraded
localization to 20.57 px. It is therefore rejected rather than selected on
detection alone; the deployed `real_v3` pair remains the safer evidence-backed
choice. This gap is consistent with the measured between-flight domain shift
and warrants more diverse real tracks before physical racing acceptance.

The flight-bias audit confirms that concern: a temporal-block classifier can
identify the source flight from appearance features at 0.466 accuracy versus
0.333 chance, and the largest standardized feature shift is 1.65. Strong and
mild label-preserving augmentation refits were tested against untouched
flight 08, but each worsened mean localization, so augmentation is configurable
but disabled by default. The next data collection must diversify tracks,
lighting, cameras, and obstacles; the gate-only real set cannot validate the
danger head.

The selected shared model's real-world classification overlays are under
`workspace/gap8_stdc_shared_real_overlays_v1`: 824 per-frame overlays, a
100-frame montage, all 12,630 predictions in CSV, and an in-crop bias report.
On the untouched in-crop flight-08 frames the overlay audit reproduces the
12.79 px mean and 6.26 px median error. Danger colors on these images are
explicitly visualization-only because the flights have no obstacle labels.

The mocap audit also shows that these captures do not establish the 2 m/s
racing requirement. Per-flight p95 translational speeds are only
0.220–0.252 m/s; isolated finite-difference samples above 2 m/s are not
sustained closed-loop validation and may include mocap noise.

## Reproduction

```bash
sbatch gap8_perception/run_stdc_train.slurm
sbatch gap8_perception/run_stdc_teacher.slurm
sbatch gap8_perception/run_stdc_distill.slurm
sbatch gap8_perception/run_stdc_evaluate.slurm
sbatch gap8_perception/run_stdc_real_eval.slurm
sbatch gap8_perception/run_stdc_qat.slurm
sbatch gap8_perception/run_stdc_dory_students.slurm
sbatch gap8_perception/run_stdc_dory_evaluate.slurm
sbatch gap8_perception/run_stdc_integer_parity.slurm
sbatch gap8_perception/run_stdc_nemo_dory_export.slurm
FREEZE_CORNER_PATH=1 sbatch gap8_perception/run_stdc_shared_dory.slurm
```

Output roots are under `/home/cchen/isaacsim-workspace/workspace/`:

- `gap8_stdc_design_v1`
- `gap8_stdc_teacher_v1`
- `gap8_stdc_distilled_v1`
- `gap8_stdc_design_v1_qat`
- `gap8_stdc_dory_pair_v1`
- `gap8_stdc_real_adapt_v2`
- `gap8_stdc_dory_pair_real_v3`
- `gap8_stdc_dory_pair_real_v3_int_parity`
- `gap8_stdc_shared_dory_frozen_real_v1`
- `gap8_stdc_shared_dory_frozen_real_v1_export`
