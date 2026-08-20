#!/usr/bin/env bash
# Lower the fresh sequential NEMO integer ONNX through DORY and generate the
# checksum-harness application consumed by run_sequential_gvsoc_checksum.slurm.
#SBATCH --job-name=gap8_sequential_dory
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${EXPORT:?set EXPORT to the completed sequential NEMO export directory}"
REPO=/home/cchen/gap8-drone-racing-perception
DORY_ROOT=${DORY_ROOT:-/home/cchen/dory}
DORY_PYTHON=${DORY_PYTHON:-/home/cchen/miniconda3/envs/dory/bin/python}

cd "$REPO"
PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9} PYTHONPATH="$DORY_ROOT" "$DORY_PYTHON" \
  "$REPO/gap8_perception/validate_dory_frontend.py" \
  --onnx "$EXPORT/integer/sequential_int.onnx" \
  --output "$EXPORT/dory_frontend_report.json" \
  --app-dir "$EXPORT/gap8_application"
