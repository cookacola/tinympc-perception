#!/usr/bin/env bash
#SBATCH --job-name=gap8_int_parity
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/gap8-drone-racing-perception
RUN=${RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_dory_pair_real_v3}
CHECKPOINT=${CHECKPOINT:-$RUN/selected.pt}
OUTPUT=${OUTPUT:-$RUN/integer_parity}
DATASET=/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k
SPLIT=$REPO/gap8_perception/configs/split_shards.json
cd "$REPO"
mkdir -p "$OUTPUT"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/export_stdc_dory_bridge.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT/bridge"
CUDA_VISIBLE_DEVICES="" \
  /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_stdc_dory_export.py \
  --bridge "$OUTPUT/bridge" --dataset "$DATASET" \
  --split-file "$SPLIT" --output "$OUTPUT/integer"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_stdc_integer_parity.py \
  --dataset "$DATASET" \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file "$SPLIT" --integer-dir "$OUTPUT/integer" \
  --output "$OUTPUT/heldout_parity_metrics.json"
