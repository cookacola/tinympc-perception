#!/usr/bin/env bash
#SBATCH --job-name=gap8_nemo_dory
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
# Old NeMO uses hash-ordered graph traversal while assigning integer
# requantization stages. Keep the validated traversal seed explicit.
export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/isaacsim-workspace
CHECKPOINT=${CHECKPOINT:-$REPO/workspace/gap8_rollout_float_baseline_rf123/best_total.pt}
DATASET=${DATASET:-$REPO/workspace/course_hm01b0_calibrated_50k}
OUTPUT=${OUTPUT:-$REPO/workspace/gap8_rollout_nemo_dory_ptq_rf123}
NANOCOCKPIT_REPO=${NANOCOCKPIT_REPO:-/home/cchen/tinympc-nanocockpit}
CONTROLLER_REPO=${CONTROLLER_REPO:-/home/cchen/tinympc-crazyflie}
VALIDATION_METRICS=${VALIDATION_METRICS:-$REPO/workspace/gap8_rollout_float_baseline_rf123/evaluation/validation_metrics.json}
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
NEMO_QAT_ARGS=()
if [[ -n "${NEMO_QAT_CHECKPOINT:-}" ]]; then
  NEMO_QAT_ARGS=(--nemo-qat-checkpoint "$NEMO_QAT_CHECKPOINT")
fi
cd "$REPO"
mkdir -p "$OUTPUT"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/export_nemo_bridge.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT/bridge"
CUDA_VISIBLE_DEVICES="" \
  /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_integer_export.py \
  --bridge "$OUTPUT/bridge" --dataset "$DATASET" --output "$OUTPUT/integer" \
  --parity-output-dir "$OUTPUT/integer_parity_samples" \
  "${NEMO_QAT_ARGS[@]}"
python3 gap8_perception/verify_integer_parity.py \
  --report "$OUTPUT/integer/nemo_export_report.json" \
  --max-corner-peak-error "${MAX_CORNER_PEAK_ERROR:-4.5}" \
  --max-danger-probability-mae "${MAX_DANGER_PROBABILITY_MAE:-0.18}"
cd /home/cchen/dory
PYTHONPATH=/home/cchen/dory \
  /home/cchen/miniconda3/envs/dory/bin/python \
  "$REPO/gap8_perception/validate_dory_frontend.py" \
  --onnx "$OUTPUT/integer/gap8_packed_int.onnx" \
  --output "$OUTPUT/dory_frontend_report.json" \
  --app-dir "$OUTPUT/gap8_application"

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
export PATH="$GAP_TOOLCHAIN/bin:$PATH"
export GAP_RISCV_GCC_TOOLCHAIN="$GAP_TOOLCHAIN"
# gap_sdk's common.sh is sourced with `set -u` active and probes this
# optional variable before it assigns GAP_RISCV_GCC_TOOLCHAIN.
export GAP_RISCV_GCC_TOOLCHAIN_BASE=""
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export BOARD_NAME=${BOARD_NAME:-gapuino}
# The vendor SDK setup scripts read several optional environment variables
# without guarding them. Temporarily relax nounset only while sourcing them.
set +u
source "$GAP_SDK/configs/gap8_v2.sh"
set -u
cd "$OUTPUT/gap8_application"
make clean
make all run platform=gvsoc CORE=7 DORY_CHECKSUM_HARNESS=1 \
  2>&1 | tee "$OUTPUT/gvsoc_checksum.log"
grep -q "Checking final layer: Checksum OK" "$OUTPUT/gvsoc_checksum.log"

cd "$REPO"
python3 gap8_perception/package_nanocockpit.py \
  --dory-app "$OUTPUT/gap8_application" \
  --nemo-report "$OUTPUT/integer/nemo_export_report.json" \
  --destination \
    "$NANOCOCKPIT_REPO/src/gap/examples/pulp-frontnet/app/networks" \
  --controller-qparams-header \
    "$CONTROLLER_REPO/apps/controller_tinympc_eigen/src/perception_model_qparams.h" \
  --validation-metrics "$VALIDATION_METRICS"

set +u
source "$GAP_SDK/configs/ai_deck.sh"
set -u
cd "$NANOCOCKPIT_REPO/src/gap/examples/pulp-frontnet"
make clean NETWORK_NAME=gap8-multitask-dory
make build NETWORK_NAME=gap8-multitask-dory \
  2>&1 | tee "$OUTPUT/nanocockpit_build.log"
test -x BUILD/GAP8_V2/GCC_RISCV/pulp_frontnet
