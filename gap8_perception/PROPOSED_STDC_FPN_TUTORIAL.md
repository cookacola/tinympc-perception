# Training and release tutorial: proposed STDC-FPN

This is the reproducible path for the proposed 160 x 120 model.

## Prerequisites

Run from the `tinympc-perception` checkout.  The synthetic corpus and targets
default to:

```text
/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k
/home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4
```

Verify target alignment and the test split before a run.  Do not create a
random per-frame split.  The committed split groups capture shards.

## 1. Run tests and a short smoke train

```bash
cd /home/cchen/gap8-drone-racing-perception
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m pytest \
  gap8_perception/tests/test_stdc_design.py \
  gap8_perception/tests/test_stdc_real_data.py -q

PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/train_stdc.py \
  --architecture proposed-fpn --dataset DATASET --targets TARGETS \
  --split-file gap8_perception/configs/split_shards.json --output /tmp/fpn-smoke \
  --epochs 1 --batch-size 4 --workers 0 --train-limit 8 --val-limit 4 --no-augment
```

## 2. Synthetic pretraining and fair baseline comparison

```bash
sbatch gap8_perception/run_proposed_stdc_fpn.slurm
```

The run saves `best_total.pt`, `best_corner.pt`, and `best_danger.pt`.  Its
held-out report compares corners and DORY-compatible 10 x 8 danger metrics
with the latest selected 160 x 120 release.  Compare like for like: use the
same test shards, crop, corner threshold, and conservative max-reduced danger
target.

## 3. Mixed real-flight adaptation

```bash
sbatch --dependency=afterok:SYNTHETIC_JOB \
  gap8_perception/run_proposed_stdc_fpn_real_adapt.slurm
```

This trains with synthetic full-task batches plus real gate-corner batches.
It trains on flight 06, selects on flight 07, and reports untouched flight 08.
Never use the real set as negative danger data: it has no obstacle labels.

## 4. INT8, DORY, and NanoCockpit acceptance

1. Freeze the selected `best_total.pt`; record SHA-256 and input/output ABI.
2. Quantize each DORY partition through NeMO using deterministic calibration
   images from the training split.
3. Check float/INT8 corner and conservative-danger parity on the held-out
   split; choose the danger threshold from that split only.
4. Run the DORY frontend and inspect the generated tiling/L1 report for every
   partition.
5. Compile and checksum under GVSOC, then package the graphs and glue code for
   NanoCockpit.  Run the package workspace/decoder audit and firmware build.
6. Copy only a passing, immutable artifact set to `releases/`, with metrics,
   reports, NanoCockpit archive, model checkpoint, manifest, and SHA-256s.

The previous shared-DORY release is the deployment fallback while any of these
checks is incomplete.
