#!/usr/bin/env bash
#SBATCH --job-name=gap8_overfit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/isaacsim-workspace
cd "$REPO"
nvidia-smi
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_multitask.py \
  --dataset workspace/course_hm01b0_nbd_texture_50k \
  --targets workspace/gap8_rollout_targets \
  --split-file gap8_perception/configs/split_shards.json \
  --output workspace/gap8_rollout_overfit_256 \
  --epochs 120 --batch-size 64 --workers 4 \
  --train-limit 256 --val-limit 64
