#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_nemo
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT:?set OUTPUT}"
REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.export_sequential_bridge \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT/bridge"
NEMO_ARGS=(
  --bridge "$OUTPUT/bridge"
  --dataset "${DATASET:-$WORKSPACE/course_hm01b0_75k_v2}"
  --split-file "${SPLIT_FILE:-gap8_perception/configs/split_shards_75k.json}"
  --output "$OUTPUT/integer"
)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_sequential_integer_export.py "${NEMO_ARGS[@]}"
