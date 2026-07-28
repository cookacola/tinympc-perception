#!/usr/bin/env bash
#SBATCH --job-name=gap8_export
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
EXPORT_ARGS=()
if [[ "${DROP_GATE_HEAD:-0}" == "1" ]]; then
  EXPORT_ARGS=(--drop-gate-head)
fi
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python gap8_perception/export_gap8.py \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" \
  "${EXPORT_ARGS[@]}"
