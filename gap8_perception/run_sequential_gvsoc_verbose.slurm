#!/usr/bin/env bash
#SBATCH --job-name=gap8_gvsoc_verbose
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err

set -euo pipefail
: "${EXPORT:?set EXPORT to the completed sequential NEMO export directory}"
GAP_SDK=${GAP_SDK:-/home/cchen/gap_sdk_dory}
GAP_TOOLCHAIN=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}
APP="$EXPORT/gap8_application_verbose"
LOG="$EXPORT/gvsoc_verbose.log"
GVSOC_TIMEOUT_S=${GVSOC_TIMEOUT_S:-60}
GVSOC_ARGS=${GVSOC_ARGS:-}

test -x "$APP/BUILD/GAP8_V2/GCC_RISCV/main"
test -f "$APP/BUILD/GAP8_V2/GCC_RISCV/gvsoc_config.json"

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

cd "$APP/BUILD/GAP8_V2/GCC_RISCV"
set +e
timeout --signal=TERM "$GVSOC_TIMEOUT_S" \
  gapy --target=gapuino_v2 --platform=gvsoc --work-dir="$PWD" \
    run --exec-prepare --exec --binary="$PWD/main" $GVSOC_ARGS 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
printf 'launcher_status=%s\n' "$status" | tee -a "$LOG"
exit "$status"
