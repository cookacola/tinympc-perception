#!/usr/bin/env bash
#SBATCH --job-name=gap8_targets
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-/home/cchen/gap8-drone-racing-perception}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
DATASET=${DATASET_ROOT:-$WORKSPACE/course_hm01b0_nbd_texture_50k}
TARGETS=${TARGETS:-$WORKSPACE/gap8_rollout_targets_rendercal_v2}
CALIBRATION=${CALIBRATION:-gap8_perception/configs/hm01b0_calibration.json}
SHARD=$(printf "shard_%09d" "$((SLURM_ARRAY_TASK_ID * 1000))")
mkdir -p "$TARGETS"
cd "$SOURCE_DIR"
PYTHONPATH="$SOURCE_DIR" /home/cchen/isaacsim-env/bin/python \
  gap8_perception/generate_targets.py "$DATASET/$SHARD" \
  --output "$TARGETS/$SHARD.npz" \
  --calibration "$CALIBRATION"
