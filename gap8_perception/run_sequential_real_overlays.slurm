#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_overlays
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT:?set OUTPUT}"
REPO=/home/cchen/gap8-drone-racing-perception
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.make_sequential_real_overlays \
  --checkpoint "$CHECKPOINT" --real-root "${REAL_ROOT:-/home/cchen/real_flight_data}" \
  --flight "${FLIGHT:-flight_08}" --output "$OUTPUT" --samples "${SAMPLES:-48}"
