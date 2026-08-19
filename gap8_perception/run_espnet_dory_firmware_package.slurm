#!/usr/bin/env bash
#SBATCH --job-name=espnet_fw_pack
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodelist=a2r-main
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
NANOCOCKPIT=/home/cchen/tinympc-nanocockpit
STUDENT=${STUDENT:-$WORKSPACE/espnet_dory_student_v2}
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_deployment}
NETWORK_NAME=${NETWORK_NAME:-gap8-espnet-dory-student-v1}
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}

for graph in encoder corner_head gate_head danger_head; do
  grep -q 'Checking final layer: Checksum OK' \
    "$OUTPUT/${graph}_gvsoc_checksum.log"
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
