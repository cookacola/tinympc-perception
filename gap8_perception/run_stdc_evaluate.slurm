#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
RUN=${RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1}
CHECKPOINT=${CHECKPOINT:-$RUN/best_total.pt}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_stdc.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN/evaluation/test_metrics.json" \
  --split test --device cuda
