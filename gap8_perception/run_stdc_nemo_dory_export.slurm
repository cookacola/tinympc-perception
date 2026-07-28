#!/usr/bin/env bash
#SBATCH --job-name=gap8_stdc_export
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/gap8-drone-racing-perception
RUN=${RUN:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_dory_pair_v1}
CHECKPOINT=${CHECKPOINT:-$RUN/selected.pt}
OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_stdc_dory_pair_v1_export}
DATASET=/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_hm01b0_v2_50k
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
cd "$REPO"
mkdir -p "$OUTPUT"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/export_stdc_dory_bridge.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT/bridge"
CUDA_VISIBLE_DEVICES="" \
  /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_stdc_dory_export.py \
  --bridge "$OUTPUT/bridge" --dataset "$DATASET" --output "$OUTPUT/integer" \
  --split-file "$REPO/gap8_perception/configs/split_shards.json"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_stdc_integer_parity.py \
  --dataset "$DATASET" \
  --targets /home/cchen/isaacsim-workspace/workspace/gap8_rollout_targets_hm01b0_v4 \
  --split-file "$REPO/gap8_perception/configs/split_shards.json" \
  --integer-dir "$OUTPUT/integer" \
  --output "$OUTPUT/integer/heldout_parity_metrics.json"

for graph in corner danger; do
  if [[ "$graph" == corner ]]; then
    expected=4800
  else
    expected=80
  fi
  cd /home/cchen/dory
  PYTHONPATH=/home/cchen/dory \
    /home/cchen/miniconda3/envs/dory/bin/python \
    "$REPO/gap8_perception/validate_dory_frontend.py" \
    --onnx "$OUTPUT/integer/$graph/${graph}_int.onnx" \
    --output "$OUTPUT/${graph}_dory_report.json" \
    --app-dir "$OUTPUT/${graph}_gap8_application" \
    --expected-output-bytes "$expected"
done

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
export PATH="$GAP_TOOLCHAIN/bin:$PATH"
export GAP_RISCV_GCC_TOOLCHAIN="$GAP_TOOLCHAIN"
export GAP_RISCV_GCC_TOOLCHAIN_BASE=""
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export BOARD_NAME=${BOARD_NAME:-gapuino}
set +u
source "$GAP_SDK/configs/gap8_v2.sh"
set -u
for graph in corner danger; do
  cd "$OUTPUT/${graph}_gap8_application"
  make clean
  make all run platform=gvsoc CORE=7 DORY_CHECKSUM_HARNESS=1 \
    2>&1 | tee "$OUTPUT/${graph}_gvsoc_checksum.log"
  grep -q "Checking final layer: Checksum OK" \
    "$OUTPUT/${graph}_gvsoc_checksum.log"
done
