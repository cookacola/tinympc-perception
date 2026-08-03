#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_teacher
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_teacher_v1}
DATASET_ROOT=${DATASET_ROOT:-/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k}
TARGET_ROOT=${TARGET_ROOT:-/home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4}
SPLIT_FILE=${SPLIT_FILE:-gap8_perception/configs/split_shards.json}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file "$SPLIT_FILE" \
  --output "$OUTPUT" --epochs "${EPOCHS:-60}" --batch-size 32 --workers 8 \
  --no-augment --privileged-teacher
