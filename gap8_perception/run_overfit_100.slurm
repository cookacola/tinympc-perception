#!/usr/bin/env bash
#SBATCH --job-name=gap8_overfit100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_nbd_texture_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_rendercal_v2}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-workspace/gap8_rollout_overfit_100_v2}
EPOCHS=${EPOCHS:-500}
TRAIN_LIMIT=${TRAIN_LIMIT:-100}
VAL_LIMIT=${VAL_LIMIT:-64}
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_multitask.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --output "$TRAIN_OUTPUT" \
  --epochs "$EPOCHS" --batch-size 32 --workers 4 \
  --train-limit "$TRAIN_LIMIT" --val-limit "$VAL_LIMIT"
