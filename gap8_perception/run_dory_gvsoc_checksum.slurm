#!/usr/bin/env bash
#SBATCH --job-name=gap8_gvsoc_check
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${EXPORT:?set EXPORT to the completed STDC DORY export directory}"
: "${GRAPH:?set GRAPH to corner, danger, encoder, corner_head, or danger_head}"
case "$GRAPH" in
  corner|danger|encoder|corner_head|danger_head) ;;
  *)
  echo "unsupported GRAPH: $GRAPH" >&2
  exit 2
  ;;
esac

GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
APP="$EXPORT/${GRAPH}_gap8_application"
LOG="$EXPORT/${GRAPH}_gvsoc_checksum_release.log"
CORE=${CORE:-8}

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
export PATH="$GAP_TOOLCHAIN/bin:$GAP_SDK/install/workstation/bin:$PATH"
export GAP_RISCV_GCC_TOOLCHAIN="$GAP_TOOLCHAIN"
export GAP_RISCV_GCC_TOOLCHAIN_BASE=""
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export BOARD_NAME=${BOARD_NAME:-gapuino}
set +u
source "$GAP_SDK/configs/gap8_v2.sh"
set -u

cd "$APP"
make clean
make all image flash run.prepare platform=gvsoc CORE="$CORE" DORY_CHECKSUM_HARNESS=1
cd BUILD/GAP8_V2/GCC_RISCV
gvsoc_launcher --config=gvsoc_config.json 2>&1 | tee "$LOG"
grep -q "Checking final layer: Checksum OK" "$LOG"
