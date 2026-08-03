#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_evaluate
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUTPUT:?set OUTPUT}"
: "${DATASET:?set DATASET}"
: "${TARGETS:?set TARGETS to regenerated sequential_fixed_normal_v1 archives}"
REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.evaluate_sequential \
  --dataset "$DATASET" \
  --targets "$TARGETS" \
  --split-file "${SPLIT_FILE:-gap8_perception/configs/split_shards_75k.json}" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --split "${SPLIT:-test}"
