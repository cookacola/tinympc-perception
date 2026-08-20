#!/usr/bin/env bash
#SBATCH --job-name=espnet_dory_pack
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --nodelist=a2r-main
#SBATCH --time=03:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
NANOCOCKPIT=/home/cchen/tinympc-nanocockpit
STUDENT=${STUDENT:-$WORKSPACE/espnet_dory_student_v2}
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_deployment}
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
NETWORK_NAME=${NETWORK_NAME:-gap8-espnet-dory-student-v1}
declare -A BYTES=(
  [encoder]=51200 [corner_head]=6400 [gate_head]=1600 [danger_head]=100
)

for graph in encoder corner_head gate_head danger_head; do
  cd /home/cchen/dory
  PYTHONPATH=/home/cchen/dory /home/cchen/miniconda3/envs/dory/bin/python \
    "$REPO/gap8_perception/validate_dory_frontend.py" \
    --onnx "$OUTPUT/integer/$graph/${graph}_int.onnx" \
    --output "$OUTPUT/${graph}_dory_report.json" \
    --app-dir "$OUTPUT/${graph}_gap8_application" \
    --expected-output-bytes "${BYTES[$graph]}"
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

PACKAGE="$NANOCOCKPIT/src/gap/examples/tiny-racer/app/networks/$NETWORK_NAME"
if [[ -e "$PACKAGE" ]]; then
  echo "Refusing to overwrite existing package: $PACKAGE" >&2
  exit 2
fi
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
