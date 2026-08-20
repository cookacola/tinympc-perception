#!/usr/bin/env bash
#SBATCH --job-name=gap8_real_adapt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
SOURCE=${SOURCE:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_distilled_v1}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_real_adapt_v1}
DATASET_ROOT=${DATASET_ROOT:-/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k}
TARGET_ROOT=${TARGET_ROOT:-/home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4}
SPLIT_FILE=${SPLIT_FILE:-gap8_perception/configs/split_shards.json}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc_real_adapt.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file "$SPLIT_FILE" \
  --real-root /home/cchen/real_flight_data \
  --checkpoint "$SOURCE/best_total.pt" \
  --output "$OUTPUT" --epochs "${EPOCHS:-20}" --batch-size 32 --workers 8 --real-weight 2.0 \
  --real-augmentation-strength "${REAL_AUGMENTATION_STRENGTH:-0.35}" \
  --real-augmentation-probability "${REAL_AUGMENTATION_PROBABILITY:-0.0}" \
  --real-train-flights "${REAL_TRAIN_FLIGHTS:-flight_06}" \
  --real-validation-flights "${REAL_VALIDATION_FLIGHTS:-flight_07}"
