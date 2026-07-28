#!/usr/bin/env bash
#SBATCH --job-name=gap8_float
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
nvidia-smi
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_nbd_texture_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_rendercal_v2}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-workspace/gap8_rollout_float_baseline_v2}
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_multitask.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --output "$TRAIN_OUTPUT" \
  --epochs 80 --batch-size 32 --workers 8 --augment \
  --high-speed-stride 4
