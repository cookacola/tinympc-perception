#!/usr/bin/env bash
#SBATCH --job-name=gap8_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_nbd_texture_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_rendercal_v2}
FLOAT_CHECKPOINT=${FLOAT_CHECKPOINT:-workspace/gap8_rollout_float_baseline_v2/best_total.pt}
QAT_OUTPUT=${QAT_OUTPUT:-workspace/gap8_rollout_qat_v2}
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_qat.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --float-checkpoint "$FLOAT_CHECKPOINT" \
  --output "$QAT_OUTPUT" \
  --epochs 20 --batch-size 32 --workers 8
