#!/usr/bin/env bash
#SBATCH --job-name=gap8_shared_repair
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

# Regenerate the three-graph shared-DORY package from one bounded signed-int8
# representation.  The output path is new; never overwrite a published release.
export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/gap8-drone-racing-perception
CHECKPOINT=${CHECKPOINT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_dory_frozen_real_v1/best_total.pt}
DATASET=${DATASET:-/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_shared_dory_frozen_real_v1_export_repaired_v1}

cd "$REPO"
mkdir -p "$OUTPUT"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/export_stdc_shared_dory_bridge.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT/bridge"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_stdc_shared_dory_export.py \
  --bridge "$OUTPUT/bridge" --dataset "$DATASET" --output "$OUTPUT/integer" \
  --split-file gap8_perception/configs/split_shards.json
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_stdc_integer_parity.py \
  --dataset "$DATASET" \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file gap8_perception/configs/split_shards.json \
  --integer-dir "$OUTPUT/integer" \
  --output "$OUTPUT/integer/heldout_parity_metrics.json"

for graph in encoder corner_head danger_head; do
  case "$graph" in
    encoder) expected=38400 ;;
    corner_head) expected=4800 ;;
    danger_head) expected=80 ;;
  esac
  cd /home/cchen/dory
  PYTHONPATH=/home/cchen/dory /home/cchen/miniconda3/envs/dory/bin/python \
    "$REPO/gap8_perception/validate_dory_frontend.py" \
    --onnx "$OUTPUT/integer/$graph/${graph}_int.onnx" \
    --output "$OUTPUT/${graph}_dory_report.json" \
    --app-dir "$OUTPUT/${graph}_gap8_application" \
    --expected-output-bytes "$expected"
done
