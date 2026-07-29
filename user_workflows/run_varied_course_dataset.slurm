#!/usr/bin/env bash
#SBATCH --job-name=course_varied
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err

set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-/home/cchen/gap8-drone-racing-perception}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
PYTHON=${PYTHON:-/home/cchen/isaacsim-env/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/course_hm01b0_varied_v1_750k}
TOTAL_FRAMES=${TOTAL_FRAMES:-750000}
SHARD_SIZE=${SHARD_SIZE:-1000}
BASE_SEED=${BASE_SEED:-12026}
GATE_ASSET_ROOT=${GATE_ASSET_ROOT:-/home/cchen/isaacsim-workspace/assets/gates}
ARRAY_WIDTH=${ARRAY_WIDTH_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-15}}
# Measured compact shards are roughly 70 KiB/frame. Keep a 10 GiB reserve
# rather than partially filling the workspace and corrupting other pipelines.
EXPECTED_KIB_PER_FRAME=${EXPECTED_KIB_PER_FRAME:-75}
RESERVE_KIB=${RESERVE_KIB:-10485760}

mkdir -p "$OUTPUT_ROOT" "$WORKSPACE/logs" "$OUTPUT_ROOT/.partial"
available_kib=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
required_kib=$(( TOTAL_FRAMES * EXPECTED_KIB_PER_FRAME + RESERVE_KIB ))
if (( available_kib < required_kib )); then
  echo "insufficient disk: available=${available_kib}KiB required=${required_kib}KiB" >&2
  echo "free space or set a larger OUTPUT_ROOT filesystem before generation" >&2
  exit 75
fi

num_shards=$(( (TOTAL_FRAMES + SHARD_SIZE - 1) / SHARD_SIZE ))
for (( shard_index=SLURM_ARRAY_TASK_ID; shard_index<num_shards; shard_index+=ARRAY_WIDTH )); do
  start=$(( shard_index * SHARD_SIZE ))
  remaining=$(( TOTAL_FRAMES - start ))
  frames=$SHARD_SIZE
  if (( remaining < SHARD_SIZE )); then
    frames=$remaining
  fi
  final_dir=$(printf "%s/shard_%09d" "$OUTPUT_ROOT" "$start")
  if [[ -f "$final_dir/_SUCCESS" ]]; then
    echo "skip completed $final_dir"
    continue
  fi
  temp_dir=$(printf "%s/.partial/shard_%09d_job_%s" "$OUTPUT_ROOT" "$start" "$SLURM_JOB_ID")
  "$PYTHON" "$SOURCE_DIR/user_workflows/generate_course_shard.py" \
    --output-dir "$temp_dir" \
    --frames "$frames" \
    --start-index "$start" \
    --seed "$BASE_SEED" \
    --width 160 --height 160 \
    --layout-family procedural_v1 \
    --gate-texture-version hm01b0_v2 \
    --gate-asset-root "$GATE_ASSET_ROOT" \
    --camera-calibration "$SOURCE_DIR/gap8_perception/configs/hm01b0_calibration.json"
  "$PYTHON" "$SOURCE_DIR/user_workflows/validate_course_shard.py" \
    --shard "$temp_dir" --expected "$frames"
  "$PYTHON" "$SOURCE_DIR/user_workflows/compact_course_shard.py" \
    --shard "$temp_dir"
  touch "$temp_dir/_SUCCESS"
  mv "$temp_dir" "$final_dir"
  echo "completed $final_dir"
done
