#!/usr/bin/env bash
#SBATCH --job-name=espnet_dory_graph
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=a2r-main
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${GRAPH:?submit with --export=ALL,GRAPH=corner_head|gate_head|danger_head}"
case "$GRAPH" in
  encoder|corner_head|gate_head|danger_head) ;;
  *) echo "Unsupported graph: $GRAPH" >&2; exit 2 ;;
esac

OUTPUT=${OUTPUT:-/home/cchen/isaacsim-workspace/workspace/espnet_dory_student_v2_deployment}
REPO=/home/cchen/gap8-drone-racing-perception
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
declare -A BYTES=(
  [encoder]=51200 [corner_head]=6400 [gate_head]=1600 [danger_head]=100
)

SCRATCH=$(mktemp -d "/tmp/espnet-dory-${GRAPH}-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
cd "$SCRATCH"
PYTHONPATH=/home/cchen/dory /home/cchen/miniconda3/envs/dory/bin/python \
  "$REPO/gap8_perception/validate_dory_frontend.py" \
  --onnx "$OUTPUT/integer/$GRAPH/${GRAPH}_int.onnx" \
  --output "$OUTPUT/${GRAPH}_dory_report.json" \
  --app-dir "$OUTPUT/${GRAPH}_gap8_application" \
  --expected-output-bytes "${BYTES[$GRAPH]}"

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

cd "$OUTPUT/${GRAPH}_gap8_application"
make clean
timeout "${GVSOC_TIMEOUT:-10m}" \
  make all run platform=gvsoc CORE=7 DORY_CHECKSUM_HARNESS=1 \
  2>&1 | tee "$OUTPUT/${GRAPH}_gvsoc_checksum.log"
! grep -q 'Checksum Failed' "$OUTPUT/${GRAPH}_gvsoc_checksum.log"
grep -q 'Checking layer: Checksum OK' "$OUTPUT/${GRAPH}_gvsoc_checksum.log"
