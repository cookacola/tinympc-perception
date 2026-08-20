#!/usr/bin/env bash
# Canonical fixed-normal targets for the fresh reality_centerline corpus only.
#SBATCH --job-name=reality_centerline_targets
#SBATCH --array=0-149%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
DATASET=${DATASET:-$WORKSPACE/reality_centerline_75k_20260731}
TARGETS=${TARGETS:-$WORKSPACE/reality_centerline_targets_75k_20260731}
SHARD=$(printf 'shard_%09d' "$((SLURM_ARRAY_TASK_ID * 500))")

test -f "$DATASET/$SHARD/_SUCCESS"
mkdir -p "$TARGETS"
STAGING=$(mktemp -d "$TARGETS/.${SHARD}.${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.generate_sequential_targets \
  "$DATASET/$SHARD" --output "$STAGING/$SHARD.npz" \
  --calibration gap8_perception/configs/hm01b0_calibration.json
mv "$STAGING/$SHARD.npz" "$TARGETS/$SHARD.npz"
mv "$STAGING/$SHARD.json" "$TARGETS/$SHARD.json"
