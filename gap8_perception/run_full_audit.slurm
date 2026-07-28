#!/usr/bin/env bash
#SBATCH --job-name=gap8_audit
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
DATASET_ROOT=${DATASET_ROOT:-workspace/course_hm01b0_calibrated_50k}
TARGET_ROOT=${TARGET_ROOT:-workspace/gap8_rollout_targets_calibrated_v3}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-workspace/gap8_rollout_audit_v3/full_dataset_audit.json}
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/full_dataset_audit.py \
  --dataset "$DATASET_ROOT" \
  --targets "$TARGET_ROOT" \
  --output "$AUDIT_OUTPUT"
