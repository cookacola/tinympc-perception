#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_real
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
RUN=${RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_design_v1}
CHECKPOINT=${CHECKPOINT:-$RUN/best_total.pt}
FLIGHTS=${FLIGHTS:-flight_06,flight_07,flight_08}
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_stdc_real.py \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN/evaluation/real_flights.json" --device cuda \
  --flights "$FLIGHTS"
