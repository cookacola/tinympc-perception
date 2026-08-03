#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_targets
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
DATASET=${DATASET:-$WORKSPACE/course_hm01b0_nbd_hm01b0_v2_50k}
TARGETS=${TARGETS:-$WORKSPACE/gap8_sequential_targets_centerline_v1}
SHARD=$(printf 'shard_%09d' "$((SLURM_ARRAY_TASK_ID * 1000))")
mkdir -p "$TARGETS"
cd "$REPO"
STAGING=$(mktemp -d "$TARGETS/.${SHARD}.${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.generate_sequential_targets \
  "$DATASET/$SHARD" --output "$STAGING/$SHARD.npz" \
  --calibration gap8_perception/configs/hm01b0_calibration.json
mv "$STAGING/$SHARD.npz" "$TARGETS/$SHARD.npz"
mv "$STAGING/$SHARD.json" "$TARGETS/$SHARD.json"
