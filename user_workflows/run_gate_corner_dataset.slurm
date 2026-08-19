#!/usr/bin/env bash
#SBATCH --job-name=gate_corners
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err

set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-${SLURM_SUBMIT_DIR:-/home/cchen/tinympc-perception}}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
PYTHON=${PYTHON:-/home/cchen/isaacsim-env/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/newbeedrone_photo_v1_corners_20k}
TOTAL_FRAMES=${TOTAL_FRAMES:-20000}
SHARD_SIZE=${SHARD_SIZE:-1000}
BASE_SEED=${BASE_SEED:-20260818}
ARRAY_WIDTH=${SLURM_ARRAY_TASK_COUNT:-4}

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/.partial" "$WORKSPACE/logs"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Python executable: $PYTHON"
"$PYTHON" --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
num_shards=$(( (TOTAL_FRAMES + SHARD_SIZE - 1) / SHARD_SIZE ))

for (( shard_index=SLURM_ARRAY_TASK_ID; shard_index<num_shards; shard_index+=ARRAY_WIDTH )); do
  start=$(( shard_index * SHARD_SIZE ))
  remaining=$(( TOTAL_FRAMES - start ))
  frames=$SHARD_SIZE
  if (( remaining < SHARD_SIZE )); then frames=$remaining; fi
  final_dir=$(printf '%s/shard_%09d' "$OUTPUT_ROOT" "$start")
  if [[ -f "$final_dir/_SUCCESS" ]]; then
    echo "skip completed $final_dir"
    continue
  fi
  if [[ -e "$final_dir" ]]; then
    echo "refusing to overwrite incomplete shard: $final_dir" >&2
    exit 2
  fi
  partial=$(printf '%s/.partial/shard_%09d_job_%s_task_%s' \
    "$OUTPUT_ROOT" "$start" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID")
  if [[ -e "$partial" ]]; then
    echo "refusing to overwrite partial shard: $partial" >&2
    exit 2
  fi

  cd "$SOURCE_DIR"
  "$PYTHON" user_workflows/generate_course_shard.py \
    --output-dir "$partial" --frames "$frames" --start-index "$start" \
    --seed "$BASE_SEED" --width 160 --height 160 --rt-subframes 1 \
    --gate-texture-version photo_v1 --gate-target-probability 0.9 \
    --camera-calibration gap8_perception/configs/hm01b0_calibration.json
  "$PYTHON" user_workflows/validate_course_shard.py \
    --shard "$partial" --expected "$frames"
  "$PYTHON" user_workflows/compact_course_shard.py --shard "$partial"
  touch "$partial/_SUCCESS"
  mv "$partial" "$final_dir"
  echo "completed $final_dir"
done
echo "Job finished at: $(date)"
