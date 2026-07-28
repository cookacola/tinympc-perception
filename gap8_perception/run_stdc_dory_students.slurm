#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_dory
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
RICH=${RICH:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1}
RICH_CHECKPOINT=${RICH_CHECKPOINT:-$RICH/best_total.pt}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_dory_pair_v1}
REAL_ARGS=()
if [[ -n "${REAL_ROOT:-}" ]]; then
  REAL_ARGS=(--real-root "$REAL_ROOT")
fi
SELECTION_ARGS=()
if [[ "${SELECT_CORNER_LAST:-0}" == "1" ]]; then
  SELECTION_ARGS=(--select-corner-last)
fi
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc_dory_students.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --teacher-checkpoint "$RICH_CHECKPOINT" \
  --output "$OUTPUT" --epochs "${EPOCHS:-20}" --batch-size 32 --workers 8 \
  "${REAL_ARGS[@]}" \
  --real-train-flights "${REAL_TRAIN_FLIGHTS:-flight_06}" \
  --real-validation-flights "${REAL_VALIDATION_FLIGHTS:-flight_07}" \
  --real-weight "${REAL_WEIGHT:-2.0}" \
  "${SELECTION_ARGS[@]}"
