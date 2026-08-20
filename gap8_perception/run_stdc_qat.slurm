#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
FLOAT_RUN=${FLOAT_RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1}
QAT_RUN=${QAT_RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1_qat}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc_qat.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --float-checkpoint "$FLOAT_RUN/best_total.pt" \
  --output "$QAT_RUN" \
  --epochs 20 --batch-size 32 --workers 8
