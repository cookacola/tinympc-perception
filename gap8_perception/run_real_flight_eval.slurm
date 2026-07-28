#!/usr/bin/env bash
#SBATCH --job-name=gap8_real_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${CHECKPOINT:=workspace/gap8_rollout_qat_v3/best_qat.pt}"
OUTPUT_DIR=${OUTPUT_DIR:-workspace/gap8_rollout_qat_v3}
OUTPUT=${OUTPUT:-$OUTPUT_DIR/real_flight_metrics.json}
VISUALIZATIONS=${VISUALIZATIONS:-$OUTPUT_DIR/real_flight_predictions}
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_real_flights.py \
  --checkpoint "$CHECKPOINT" \
  --real-root /home/cchen/real_flight_data \
  --output "$OUTPUT" \
  --visualizations "$VISUALIZATIONS" \
  --label-tolerance-px 3.25
