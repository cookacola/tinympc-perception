#!/usr/bin/env bash
# Regenerate the 75k target archives using the current sequential-label schema.
# The existing archives are left in place until a replacement archive has been
# generated successfully for its shard.
#SBATCH --job-name=gap8_sequential_targets_75k
#SBATCH --array=0-149%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
DATASET=${DATASET:?set DATASET}
TARGETS=${TARGETS:?set TARGETS}
SHARD=$(printf 'shard_%09d' "$((SLURM_ARRAY_TASK_ID * 500))")

cd "$REPO"
mkdir -p "$TARGETS"
STAGING=$(mktemp -d "$TARGETS/.${SHARD}.${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT

PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.generate_sequential_targets \
  "$DATASET/$SHARD" \
  --output "$STAGING/$SHARD.npz" \
  --calibration gap8_perception/configs/hm01b0_calibration.json

mv "$STAGING/$SHARD.npz" "$TARGETS/$SHARD.npz"
mv "$STAGING/$SHARD.json" "$TARGETS/$SHARD.json"
