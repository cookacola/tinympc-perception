#!/usr/bin/env bash
#SBATCH --job-name=gap8_pipeline_audit
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${DATASET_ROOT:?set DATASET_ROOT}"
: "${TARGET_ROOT:?set TARGET_ROOT}"
: "${AUDIT_ROOT:?set AUDIT_ROOT}"
mkdir -p "$AUDIT_ROOT"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/full_dataset_audit.py \
  --dataset "$DATASET_ROOT" --targets "$TARGET_ROOT" \
  --output "$AUDIT_ROOT/full_dataset_audit.json"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/summarize_targets.py \
  --targets "$TARGET_ROOT" \
  --split-file gap8_perception/configs/split_shards.json \
  --output "$AUDIT_ROOT/target_summary.json"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/inspect_dataset.py "$DATASET_ROOT" \
  --output "$AUDIT_ROOT" --samples 100 --targets "$TARGET_ROOT"
