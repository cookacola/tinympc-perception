#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_distill
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
STUDENT=${STUDENT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1}
TEACHER=${TEACHER:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_teacher_v1}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_distilled_v1}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc_distill.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --student-checkpoint "$STUDENT/best_total.pt" \
  --teacher-checkpoint "$TEACHER/best_total.pt" \
  --output "$OUTPUT" --epochs 20 --batch-size 32 --workers 8 --alpha 0.35
