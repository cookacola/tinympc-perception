#!/usr/bin/env bash
#SBATCH --job-name=light_cal
#SBATCH --array=0-4%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
ROOT=${ROOT:-$WORKSPACE/newbeedrone_lighting_calibration_v1}
TEXTURES=${TEXTURES:-$WORKSPACE/commons_clutter_textures_v1/manifest.json}
scales=(1.0 1.5 2.0 2.5 3.0)
scale=${scales[$SLURM_ARRAY_TASK_ID]}
output=$(printf '%s/scale_%s' "$ROOT" "${scale/./p}")

mkdir -p "$ROOT" "$WORKSPACE/logs"
if [[ -e "$output" ]]; then echo "refusing to overwrite $output" >&2; exit 2; fi
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID scale: $scale node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"
nvidia-smi
cd "$SOURCE"
python user_workflows/generate_course_shard.py \
  --output-dir "$output" --frames 140 --start-index 0 --seed 20260819 \
  --width 160 --height 160 --rt-subframes 1 --lighting-scale "$scale" \
  --gate-texture-version photo_v1 --gate-target-probability 0.85 \
  --layout-family multi_course_v2 --asset-split train \
  --clutter-texture-manifest "$TEXTURES" \
  --camera-calibration gap8_perception/configs/hm01b0_calibration.json
python user_workflows/validate_course_shard.py --shard "$output" --expected 140
