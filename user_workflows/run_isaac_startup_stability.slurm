#!/usr/bin/env bash
#SBATCH --job-name=isaac_startup
#SBATCH --array=0-4%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=00:20:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=/home/cchen/tinympc-gate-texture
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
ROOT=$WORKSPACE/isaac_startup_stability_v1
output=$(printf '%s/run_%02d' "$ROOT" "$SLURM_ARRAY_TASK_ID")
mkdir -p "$ROOT" "$WORKSPACE/logs"
if [[ -e "$output" ]]; then echo "refusing to overwrite $output" >&2; exit 2; fi
cd "$SOURCE"
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID node: $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
timeout --signal=TERM --kill-after=20s 120s \
  python user_workflows/generate_course_shard.py \
    --output-dir "$output" --frames 7 --start-index "$((SLURM_ARRAY_TASK_ID * 7))" \
    --seed 20260819 --width 160 --height 160 --rt-subframes 1 --lighting-scale 3.0 \
    --gate-texture-version photo_v1 --gate-target-probability 0.85 \
    --layout-family multi_course_v2 --asset-split train \
    --clutter-texture-manifest "$WORKSPACE/commons_clutter_textures_v1/manifest.json" \
    --camera-calibration gap8_perception/configs/hm01b0_calibration.json
python user_workflows/validate_course_shard.py --shard "$output" --expected 7
