# GAP8 multi-task drone-racing perception

This project trains a 160x160 monochrome, GAP8-oriented CNN with:

- four ordered 40x40 corner heatmaps: top-left, top-right, bottom-right,
  bottom-left;
- one 20x20 nominal-speed collision-within-horizon map;
- 20x20 conservative inverse-range and uncertainty maps;
- an optional removable 40x40 gate-opening head.

The stock DORY graph has one image input. It predicts collision risk for the
documented nominal 1.0 m/s state, conservative inverse range, and uncertainty.
The controller-facing
postprocessor in `danger_postprocessor.py` combines those outputs with current
speed, MPC horizon, and perception/control latency to produce the final
20x20 speed-dependent danger probability and TTC. This implements the permitted
nominal-target/controller-postprocessor alternative. The controller shifts
the nominal collision logit using live motion and uses range reachability as
a conservative high-speed fallback.

The deployed network has one packed 40x40 output tensor:
`[TL,TR,BR,BL,nominal_collision,inverse_range,uncertainty,gate]`. The wire
field retains the historical `obstacle_presence` name for ABI compatibility.
GAP8 splits the channels and conservatively max-pools collision, range, and
uncertainty to 20x20; gate opening is min-pooled because it grants permission.
The STM32 runs the range/state danger postprocessor. Packing is intentional:
stock DORY supports a single feed-
forward output graph with residuals, not arbitrary multi-head branches. It
also avoids ONNX `Resize`.

## Dataset and limitations

The corrected source is
`workspace/course_hm01b0_nbd_hm01b0_v2_50k`. It contains 50
successful 1,000-frame shards with aligned RGB, HM01B0-like monochrome,
millimeter depth, class semantics, and random camera eye/look-at poses.

The source does **not** contain gate instance IDs, authoritative gate corners,
per-frame gate poses, vehicle velocity/attitude, timestamps, or trajectories.
It also contains one fixed scene. Consequently:

- gate openings remain raster-derived from closed semantic holes; corner
  regression additionally excludes openings below 100 px², with a side below
  5 px, or with a side ratio above 6, so distant/edge-on gates do not create a
  fictitious localization error;
- each image is paired with synthetic 0.5, 1.0, and 5.0 m/s body-forward
  states because source motion was not recorded;
- rollout danger is exact for the reconstructed fixed-scene AABBs, but the
  state distribution and collision geometry still require real/closed-loop
  validation;
- the 40k/5k/5k split is grouped by capture shard/generator seed, but cannot
  measure new-scene generalization;
- PnP and closed-loop collision metrics are reported unavailable rather than
  fabricated.

The deployed HM01B0 calibration is in `configs/hm01b0_calibration.json`.
The supplied OpenCV matrix and five distortion coefficients are used by the
controller interface. Average calibration reprojection error is 0.1573 px.
Known gate geometry revealed that the legacy renders used Isaac's default
`fx≈fy≈183.25 px`, not the supplied `fx≈89 px` camera. The corrected v4 render
at `workspace/course_hm01b0_nbd_hm01b0_v2_50k` applies the supplied OpenCV
camera model and NewBeeDrone texture. See `PIPELINE_STATUS.md`.

### Real-flight verification

`/home/cchen/real_flight_data` contains 23,288 real 160x160 HM01B0 frames from
three flights, including 12,630 positive gate frames with mocap-derived corner
labels. These flights contain gates but no obstacles. They are therefore used
only for held-out gate/corner sim-to-real verification and camera-domain
inspection, not for danger, range, collision, or false-safe claims.

The raw corner sequence changes with viewing side even though the metadata
names one fixed order. `audit_real_flights.py` canonicalizes every quadrilateral
to image-plane TL,TR,BR,BL. The supplied coordinates are also treated as noisy:
the 100-frame visual audit and full diagnostic report are under
`workspace/real_flight_audit`. The current nearest-edge diagnostic is median
0.96 px and p95 3.25 px; mocap/image nearest-timestamp difference is p95
4.03 ms. `evaluate_real_flights.py` reports raw model error and a separately
identified tolerance-adjusted bound, grouped by whole flight. It does not
silently use unlabeled frames as obstacle-free danger supervision.

## Danger assumptions

- calibrated center ray for each 20x20 cell;
- candidate desired velocity toward that ray;
- 1.0 s horizon at 50 ms rollout steps after 80 ms latency;
- 0.10 m drone radius plus 0.10 m safety margin;
- acceleration limited to 6 m/s² and by a 35-degree attitude bound;
- swept-sphere collision checking against floor, course boundary, gate frames,
  obstacles, and lab clutter AABBs;
- primary binary collision-within-horizon target, plus minimum clearance, TTC,
  normalized inverse TTC, and boundary-proximity uncertainty.
- the image head is trained at the nominal 1.0 m/s state; validation/test keep
  0.5/1/5 m/s variants and evaluate the live-state postprocessor.

## Inspect and generate targets

```bash
cd /home/cchen/isaacsim-workspace
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/inspect_dataset.py \
  workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --output workspace/gap8_rollout_audit_hm01b0_v4 --samples 100

PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/make_split.py \
  workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --output gap8_perception/configs/split_shards.json

sbatch --export=ALL,DATASET_ROOT=workspace/course_hm01b0_nbd_hm01b0_v2_50k,\
TARGETS=workspace/gap8_rollout_targets_hm01b0_v4,\
CALIBRATION=gap8_perception/configs/hm01b0_calibration.json \
  gap8_perception/run_targets.slurm
```

## Tests and Stage A overfit gate

```bash
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m pytest -q \
  gap8_perception/tests
sbatch gap8_perception/run_overfit_100.slurm
```

Full training must not start unless corner, danger, and gate training losses
all become very low on the 100-image memorization run.

## Float baseline and ablation

```bash
sbatch gap8_perception/run_float_baseline.slurm
sbatch gap8_perception/run_float_no_gate.slurm
```

The baseline saves `last.pt`, `best_total.pt`, `best_corner.pt`, and
`best_danger.pt`. Resume with:

```bash
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_multitask.py ... \
  --resume workspace/gap8_rollout_float_hm01b0_v4/last.pt
```

Evaluate and visualize:

```bash
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/evaluate.py \
  --dataset workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint workspace/gap8_rollout_float_hm01b0_v4/best_total.pt \
  --split test \
  --output workspace/gap8_rollout_float_hm01b0_v4/evaluation/test_metrics.json \
  --evaluation-calibration gap8_perception/configs/hm01b0_calibration.json

PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/visualize_predictions.py \
  --dataset workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint workspace/gap8_rollout_float_hm01b0_v4/best_total.pt \
  --output workspace/gap8_rollout_float_hm01b0_v4/evaluation/predictions
```

## QAT and DORY/GAP8 export

```bash
sbatch gap8_perception/run_qat.slurm

NEMO_GRAPH_HASH_SEED=9 sbatch --export=ALL,NEMO_GRAPH_HASH_SEED,\
CHECKPOINT=workspace/gap8_rollout_float_hm01b0_v4/best_total.pt,\
DATASET=workspace/course_hm01b0_nbd_hm01b0_v2_50k,\
NEMO_QAT_CHECKPOINT=workspace/gap8_nemo_distill24_hm01b0_v4/epoch_22_nemo_qat.pt,\
OUTPUT=workspace/gap8_rollout_nemo_dory_hm01b0_v4_epoch22_seed9 \
  gap8_perception/run_nemo_dory_export.slurm
```

The exporter folds BatchNorm and rejects operators outside the installed DORY
Quantlab frontend's inference surface. The exported single-input cluster graph
contains only Conv/ReLU/Add plus ignorable Identity nodes; specifically, it
contains no Resize.

Passing the structural ONNX check is not the deployment finish line.
DORY expects a Quantlab/NEMO-style integer graph with its recognized
requantization patterns. Required final gates are: generate that graph, parse
it with the local DORY frontend, generate and compile the GAP8 application,
run numerical parity in GVSOC or on hardware, and measure L1/L2/L3 usage and
latency on the physical GAP8.

`run_nemo_dory_export.slurm` performs the NEMO/DORY toolchain conversion.
The current package starts from the selected float checkpoint and uses the
NeMO-native epoch-22 distillation checkpoint:

1. bridge the modern checkpoint to a version-neutral NPZ;
2. calibrate and integerize it with the installed NEMO/PyTorch-1.4
   environment while explicitly hiding CUDA;
3. export the NEMO integer ONNX;
4. require the installed DORY NEMO frontend and GAP8 backend tiler to pass.

The graph maps to 48 fused GAP8 hardware nodes, 11.904 M MACs, and an
estimated maximum L1 tile of 36,240 bytes against the 64,000-byte DORY target.
The exact 32-image integer comparison is substantially worse than the float
teacher. That loss has been explicitly accepted for obstacle-avoidance
integration and is documented as deployment risk. See `PIPELINE_STATUS.md`
and `NANOCOCKPIT_DEPLOYMENT.md` for measured parity, runtime safeguards, and
remaining validation work.

## Controller interface

`controller_interface.py` decodes corner confidence, computes `S=G*(1-D)`,
erodes the selected connected corridor by a configurable image margin, fits
image boundary lines, undistorts them, and rotates camera-centered angular
plane normals into the TinyMPC frame. For the preferred look-ahead mode it
returns rows

`-n_w^T p_k - tau_k n_w^T v_k - s_k <= -n_w^T p_c0`.

It also supports velocity-only rows, explicit position/velocity/slack state
indices, and slack diagnostics. Planes are frozen during one solve and rebuilt
at every perception frame. A metric obstacle-plane helper refuses to construct
an offset without a metric boundary point or conservative range.

The required A–F closed-loop protocol is recorded in
`configs/controller_ablations.json`. Actual collision/completion/slack/solve
time results require integration with the flight TinyMPC loop; they are not
inferred from offline images.

For a requirement-by-requirement distinction between completed software,
failed acceptance gates, and hardware-only work, see `COMPLETION_AUDIT.md`.
