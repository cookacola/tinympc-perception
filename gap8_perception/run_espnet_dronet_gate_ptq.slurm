#!/usr/bin/env bash
#SBATCH --job-name=esp-dronet-gate-ptq
#SBATCH --nodelist=a2r-main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
repo=/home/cchen/gap8-drone-racing-perception
workspace=/home/cchen/isaacsim-workspace/workspace
bridge=${BRIDGE:?set BRIDGE to an exported shared-filesystem bridge}
output=${OUTPUT:?set OUTPUT to a new deployment directory}
dataset=${DATASET:-$workspace/tinympc_event_aligned_dataset_20260816}
dory_root=${DORY_ROOT:-/home/cchen/dory}
gap_sdk=${GAP_SDK:-/home/cchen/gap_sdk_dory}
gap_toolchain=${GAP_TOOLCHAIN:-/home/cchen/gap_riscv_toolchain_ubuntu_18}

if ! grep -q 'weights_name = None' "$dory_root/dory/Frontend_frameworks/NEMO/Parser.py"; then
  echo "Apply gap8_perception/patches/dory-nemo-pooling-data-layout.patch to $dory_root first" >&2
  exit 2
fi
if ! grep -q 'previous_l2_output is not None' "$dory_root/dory/Hardware_targets/PULP/Common/Tiler/tiler_pool2d.py"; then
  echo "Apply gap8_perception/patches/dory-nemo-pooling-data-layout.patch to $dory_root first" >&2
  exit 2
fi

mkdir -p "$output"
cd "$repo"
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"
qat_args=()
if [[ -n "${QAT_DIRECTORY:-}" ]]; then
  qat_args=(--qat-directory "$QAT_DIRECTORY")
fi
canonical_args=()
if [[ "${CANONICALIZE_BATCHNORM:-0}" == 1 ]]; then
  canonical_args+=(--canonicalize-batchnorm)
fi
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python \
  gap8_perception/nemo_espnet_dronet_gate_export.py \
  --bridge "$bridge" --dataset "$dataset" --output "$output/integer" \
  --signed-weight-bits "${SIGNED_WEIGHT_BITS:-7}" \
  "${qat_args[@]}" "${canonical_args[@]}"

declare -A bytes=(
  [encoder]=25600 [corner_head]=1600 [gate_head]=400
  [presence_head]=1 [navigation_head]=2
)
for graph in encoder corner_head gate_head presence_head navigation_head; do
  cd "$dory_root"
  PYTHONPATH="$dory_root" /home/cchen/miniconda3/envs/dory/bin/python \
    "$repo/gap8_perception/validate_dory_frontend.py" \
    --onnx "$output/integer/$graph/${graph}_int.onnx" \
    --output "$output/${graph}_dory_report.json" \
    --app-dir "$output/${graph}_gap8_application" \
    --expected-output-bytes "${bytes[$graph]}" --l2-buffer-bytes 300000
done

export PATH="$gap_toolchain/bin:$PATH"
export GAP_RISCV_GCC_TOOLCHAIN="$gap_toolchain"
export GAP_RISCV_GCC_TOOLCHAIN_BASE=''
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export BOARD_NAME=${BOARD_NAME:-gapuino}
set +u
source "$gap_sdk/configs/gap8_v2.sh"
set -u
for graph in encoder corner_head gate_head presence_head navigation_head; do
  cd "$output/${graph}_gap8_application"
  make clean
  make all run platform=gvsoc CORE=7 DORY_CHECKSUM_HARNESS=1 \
    2>&1 | tee "$output/${graph}_gvsoc_checksum.log"
  if grep -q 'Checksum Failed' "$output/${graph}_gvsoc_checksum.log"; then
    echo "GVSOC checksum mismatch in $graph" >&2
    exit 3
  fi
  grep -q 'Checking layer: Checksum OK' "$output/${graph}_gvsoc_checksum.log"
done
