#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
TEACHER=${TEACHER:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_real_adapt_v2/best_total.pt}
INITIAL_PAIR=${INITIAL_PAIR:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_dory_pair_real_v3/selected.pt}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_dory_real_v1}
cd "$REPO"
freeze_args=()
if [[ "${FREEZE_CORNER_PATH:-0}" == "1" ]]; then
  freeze_args+=(--freeze-corner-path)
fi
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc_shared_dory.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --teacher-checkpoint "$TEACHER" \
  --initial-pair-checkpoint "$INITIAL_PAIR" \
  --real-root /home/cchen/real_flight_data \
  --output "$OUTPUT" --epochs "${EPOCHS:-20}" --batch-size 32 --workers 8 \
  "${freeze_args[@]}" \
  --real-weight "${REAL_WEIGHT:-2.0}" \
  --real-train-flights "${REAL_TRAIN_FLIGHTS:-flight_06}" \
  --real-validation-flights "${REAL_VALIDATION_FLIGHTS:-flight_07}"
