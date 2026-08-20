#!/usr/bin/env bash
#SBATCH --job-name=gap8_shared_canonical_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

# The pinned Frontnet NeMO environment uses PyTorch 1.4/CUDA 10.1, which does
# not have kernels for Ada/Blackwell GPUs. Keep this QAT job CPU-only; DORY and
# the release ABI remain pinned and reproducible.
REPO=/home/cchen/gap8-drone-racing-perception
BRIDGE=${BRIDGE:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_dory_frozen_real_v1_export_repaired_v1/bridge}
DATASET=${DATASET:-/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_canonical_qat_v1}
EPOCHS=${EPOCHS:-12}

mkdir -p "$OUTPUT"
cd "$REPO"
for graph in encoder corner_head danger_head; do
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/cchen/miniconda3/envs/frontnet/bin/python \
    gap8_perception/train_nemo_shared_dory_qat.py \
    --graph "$graph" --bridge "$BRIDGE" --dataset "$DATASET" \
    --output "$OUTPUT" --epochs "$EPOCHS" --batch-size 32 --workers 8 \
    --residual-requantization-factor 1
done
