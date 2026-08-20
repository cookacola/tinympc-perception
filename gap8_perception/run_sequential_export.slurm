#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_export
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT to a SequentialSTDCNet checkpoint}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
cd /home/cchen/gap8-drone-racing-perception
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.export_sequential_onnx \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR"
