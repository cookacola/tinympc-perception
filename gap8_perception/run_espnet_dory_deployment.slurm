#!/usr/bin/env bash
#SBATCH --job-name=espnet_nemo_dory
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=a2r-main
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/gap8-drone-racing-perception
TRAINING_REPO=/home/cchen/tinympc-gate-texture
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
NANOCOCKPIT=/home/cchen/tinympc-nanocockpit
STUDENT=${STUDENT:-$WORKSPACE/espnet_dory_student_v2}
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_deployment}
DATASET=$WORKSPACE/tinympc_event_aligned_dataset_20260816
GATE_DATASET=$WORKSPACE/newbeedrone_multi_course_gate_v2_20k
GATE_TARGETS=$WORKSPACE/newbeedrone_multi_course_gate_v2_targets
NO_GATE=$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k
PAIRED_NO_GATE=$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k
TEACHER=${TEACHER:-$WORKSPACE/newbeedrone_multi_course_safety_constrained_v3/selected_for_deployment.pt}
OBSTACLE_CHECKPOINT=$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt
GATE_CHECKPOINT=$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
DORY_ROOT=${DORY_ROOT:-/home/cchen/dory}
NETWORK_NAME=${NETWORK_NAME:-gap8-espnet-dory-student-v2}

if ! grep -q 'uint16_t add_shift' \
  "$DORY_ROOT/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/64bit/include/pulp_nn_kernels.h"; then
  echo "Apply gap8_perception/patches/dory-gap8-exact-residual-requant.patch to $DORY_ROOT first" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
cd "$TRAINING_REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/export_espnet_dory_student_bridge.py \
  --checkpoint "$STUDENT/best.pt" --output "$OUTPUT/bridge"

cd "$REPO"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_espnet_dory_student_export.py \
  --bridge "$OUTPUT/bridge" --dataset "$DATASET" --output "$OUTPUT/integer"

cd "$TRAINING_REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_espnet_dory_onnx.py \
  --integer-dir "$OUTPUT/integer" \
  --nemo-report "$OUTPUT/integer/nemo_espnet_dory_student_report.json" \
  --obstacle-dataset "$DATASET" \
  --gate-dataset "$GATE_DATASET" --gate-targets "$GATE_TARGETS" \
  --gate-split-file "$GATE_TARGETS/split_shards.json" \
  --no-gate-dataset "$NO_GATE" --paired-no-gate-dataset "$PAIRED_NO_GATE" \
  --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$OBSTACLE_CHECKPOINT" \
  --gate-checkpoint "$GATE_CHECKPOINT" --teacher-checkpoint "$TEACHER" \
  --output "$OUTPUT/integer_ground_truth.json"

declare -A BYTES=(
  [encoder]=51200 [corner_head]=6400 [gate_head]=1600 [danger_head]=100
)
declare -A L2_BYTES=(
  [encoder]=260000 [corner_head]=180000 [gate_head]=180000 [danger_head]=180000
)
for graph in encoder corner_head gate_head danger_head; do
  cd "$DORY_ROOT"
  PYTHONPATH="$DORY_ROOT" /home/cchen/miniconda3/envs/dory/bin/python \
    "$REPO/gap8_perception/validate_dory_frontend.py" \
    --onnx "$OUTPUT/integer/$graph/${graph}_int.onnx" \
    --output "$OUTPUT/${graph}_dory_report.json" \
    --app-dir "$OUTPUT/${graph}_gap8_application" \
    --expected-output-bytes "${BYTES[$graph]}" \
    --l2-buffer-bytes "${L2_BYTES[$graph]}"
done

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
export PATH="$GAP_TOOLCHAIN/bin:$PATH"
export GAP_RISCV_GCC_TOOLCHAIN="$GAP_TOOLCHAIN"
export GAP_RISCV_GCC_TOOLCHAIN_BASE=''
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export BOARD_NAME=${BOARD_NAME:-gapuino}
set +u
source "$GAP_SDK/configs/gap8_v2.sh"
set -u
for graph in encoder corner_head gate_head danger_head; do
  cd "$OUTPUT/${graph}_gap8_application"
  make clean
  make all run platform=gvsoc CORE=7 DORY_CHECKSUM_HARNESS=1 \
    2>&1 | tee "$OUTPUT/${graph}_gvsoc_checksum.log"
  ! grep -q 'Checksum Failed' "$OUTPUT/${graph}_gvsoc_checksum.log"
  grep -q 'Checking layer: Checksum OK' "$OUTPUT/${graph}_gvsoc_checksum.log"
done

cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/package_espnet_dory_nanocockpit.py \
  --encoder-app "$OUTPUT/encoder_gap8_application" \
  --corner-head-app "$OUTPUT/corner_head_gap8_application" \
  --gate-head-app "$OUTPUT/gate_head_gap8_application" \
  --danger-head-app "$OUTPUT/danger_head_gap8_application" \
  --nemo-report "$OUTPUT/integer/nemo_espnet_dory_student_report.json" \
  --integer-evaluation "$OUTPUT/integer_ground_truth.json" \
  --student-summary "$STUDENT/summary.json" \
  --destination "$NANOCOCKPIT/src/gap/examples/tiny-racer/app/networks" \
  --name "$NETWORK_NAME"

set +u
source "$GAP_SDK/configs/ai_deck.sh"
set -u
cd "$NANOCOCKPIT/src/gap/examples/tiny-racer"
make clean NETWORK_NAME="$NETWORK_NAME"
make build NETWORK_NAME="$NETWORK_NAME" \
  2>&1 | tee "$OUTPUT/nanocockpit_firmware_build.log"
test -x BUILD/GAP8_V2/GCC_RISCV/pulp_frontnet
