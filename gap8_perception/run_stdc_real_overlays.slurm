#!/usr/bin/env bash
#SBATCH --job-name=gap8_overlays
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
CHECKPOINT=${CHECKPOINT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_real_adapt_v2/best_total.pt}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_real_overlays_adapt_v2}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/visualize_stdc_real_classifications.py \
  --checkpoint "$CHECKPOINT" \
  --real-root /home/cchen/real_flight_data \
  --output "$OUTPUT" --batch-size 128 --stride 10 --device cuda
