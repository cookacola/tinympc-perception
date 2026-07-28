#!/usr/bin/env bash
#SBATCH --job-name=gap8_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_nbd_texture_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_rendercal_v2}
EVALUATION_CALIBRATION=${EVALUATION_CALIBRATION:-gap8_perception/configs/hm01b0_calibration.json}
mkdir -p "$OUTPUT_DIR"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/evaluate.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint "$CHECKPOINT" --split validation \
  --output "$OUTPUT_DIR/validation_metrics.json" --workers 8 --device cpu \
  --evaluation-calibration "$EVALUATION_CALIBRATION"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/evaluate.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint "$CHECKPOINT" --split test \
  --output "$OUTPUT_DIR/test_metrics.json" --workers 8 --device cpu \
  --evaluation-calibration "$EVALUATION_CALIBRATION"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/visualize_predictions.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint "$CHECKPOINT" --split test \
  --output "$OUTPUT_DIR/predictions" --count 16 --device cpu
