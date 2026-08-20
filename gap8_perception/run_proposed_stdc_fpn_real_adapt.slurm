#!/usr/bin/env bash
#SBATCH --job-name=gap8_proposed_real
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
REPO=/home/cchen/gap8-drone-racing-perception
DATASET=/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k
TARGETS=/home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4
SOURCE=${SOURCE:-/home/cchen/isaacsim-workspace/workspace/gap8_proposed_stdc_fpn_v1}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_proposed_stdc_fpn_real_v1}
BASELINE=/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_dory_frozen_real_v1/test_metrics.json
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/train_stdc_real_adapt.py \
  --dataset "$DATASET" --targets "$TARGETS" \
  --split-file gap8_perception/configs/split_shards.json \
  --real-root /home/cchen/real_flight_data --checkpoint "$SOURCE/best_total.pt" \
  --output "$OUTPUT" --epochs "${EPOCHS:-20}" --batch-size 32 --workers 8 \
  --real-weight 2.0 --real-train-flights flight_06 \
  --real-validation-flights flight_07
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/evaluate_stdc.py \
  --dataset "$DATASET" --targets "$TARGETS" \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint "$OUTPUT/best_total.pt" --output "$OUTPUT/test_metrics.json" \
  --split test --batch-size 128 --workers 8 --device cuda --baseline-report "$BASELINE"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/evaluate_stdc_real.py \
  --checkpoint "$OUTPUT/best_total.pt" --real-root /home/cchen/real_flight_data \
  --flights flight_08 --output "$OUTPUT/real_flight08.json" --device cuda
