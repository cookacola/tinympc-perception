#!/usr/bin/env bash
#SBATCH --job-name=gap8_overfit_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_nbd_texture_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_rendercal_v2}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-workspace/gap8_rollout_overfit_100_v2}
EPOCHS=${EPOCHS:-500}
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_multitask.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --output "$TRAIN_OUTPUT" \
  --epochs "$EPOCHS" --batch-size 32 --workers 4 \
  --train-limit 100 --val-limit 64 \
  "${RESUME_ARGS[@]}"
