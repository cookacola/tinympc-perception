#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_teacher
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_teacher_v1}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_stdc.py \
  --dataset /home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --output "$OUTPUT" --epochs 60 --batch-size 32 --workers 8 \
  --no-augment --privileged-teacher
