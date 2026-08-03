#!/usr/bin/env bash
# Held-out real-flight gate-corner evaluation for a canonical sequential run.
#SBATCH --job-name=gap8_sequential_real_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT:?set OUTPUT}"
REPO=/home/cchen/gap8-drone-racing-perception
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.evaluate_sequential_real \
  --checkpoint "$CHECKPOINT" --real-root "${REAL_ROOT:-/home/cchen/real_flight_data}" \
  --flights "${FLIGHTS:-flight_08}" --output "$OUTPUT" --device cuda
