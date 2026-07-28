#!/usr/bin/env bash
#SBATCH --job-name=course_audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err

set -euo pipefail
REPO_DIR="/home/cchen/isaacsim-workspace"
DATASET="${DATASET:-$REPO_DIR/workspace/course_hm01b0_nbd_texture_50k}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-50000}"

/home/cchen/isaacsim-env/bin/python \
    "$REPO_DIR/user_workflows/audit_course_dataset.py" \
    "$DATASET" \
    --expected-frames "$EXPECTED_FRAMES"
