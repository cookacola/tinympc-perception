#!/usr/bin/env bash
#SBATCH --job-name=gate_tex_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --exclude=a2r-main
#SBATCH --time=00:40:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-${SLURM_SUBMIT_DIR:-/home/cchen/tinympc-perception}}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
TEXTURE_VERSION=${TEXTURE_VERSION:-photo_v1}
OUTPUT=${OUTPUT:-$WORKSPACE/course_gate_texture_${TEXTURE_VERSION}_smoke_$SLURM_JOB_ID}
PARTIAL=${OUTPUT}.partial
PYTHON=/home/cchen/isaacsim-env/bin/python

cd "$SOURCE_DIR"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python executable: $PYTHON"
"$PYTHON" --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
if [[ -e "$PARTIAL" || -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing smoke output: $OUTPUT" >&2
  exit 2
fi
"$PYTHON" user_workflows/generate_course_shard.py \
  --output-dir "$PARTIAL" --frames 100 --start-index 0 --seed 20260727 \
  --rt-subframes 8 \
  --gate-target-probability 0.8 \
  --gate-texture-version "$TEXTURE_VERSION" \
  --camera-calibration gap8_perception/configs/hm01b0_calibration.json
"$PYTHON" user_workflows/validate_course_shard.py \
  --shard "$PARTIAL" --expected 100
"$PYTHON" user_workflows/compact_course_shard.py --shard "$PARTIAL"
touch "$PARTIAL/_SUCCESS"
mv "$PARTIAL" "$OUTPUT"
echo "smoke_output=$OUTPUT"
echo "Job finished at: $(date)"
