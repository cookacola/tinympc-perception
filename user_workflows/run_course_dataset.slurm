#!/usr/bin/env bash
#SBATCH --job-name=course_sdg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR="/home/cchen/isaacsim-workspace"
PYTHON="/home/cchen/isaacsim-env/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/workspace/course_dataset_50k}"
TOTAL_FRAMES="${TOTAL_FRAMES:-50000}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
WIDTH="${WIDTH:-160}"
HEIGHT="${HEIGHT:-160}"
BASE_SEED="${BASE_SEED:-2026}"
GATE_TEXTURE_VERSION="${GATE_TEXTURE_VERSION:-hm01b0_v2}"
ARRAY_WIDTH="${ARRAY_WIDTH_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-3}}"

mkdir -p "$OUTPUT_ROOT" "$REPO_DIR/workspace/logs" "$OUTPUT_ROOT/.partial"

echo "job_id=$SLURM_JOB_ID task_id=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

NUM_SHARDS=$(( (TOTAL_FRAMES + SHARD_SIZE - 1) / SHARD_SIZE ))
for (( shard_index=SLURM_ARRAY_TASK_ID; shard_index<NUM_SHARDS; shard_index+=ARRAY_WIDTH )); do
    start=$(( shard_index * SHARD_SIZE ))
    remaining=$(( TOTAL_FRAMES - start ))
    frames=$SHARD_SIZE
    if (( remaining < SHARD_SIZE )); then
        frames=$remaining
    fi
    final_dir=$(printf "%s/shard_%09d" "$OUTPUT_ROOT" "$start")
    success_file="$final_dir/_SUCCESS"
    if [[ -f "$success_file" ]]; then
        echo "skip completed $final_dir"
        continue
    fi

    available_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
    if (( available_kb < 15728640 )); then
        echo "refusing to continue: less than 15 GiB free" >&2
        exit 75
    fi

    temp_base=$(printf "%s/.partial/shard_%09d_job_%s" "$OUTPUT_ROOT" "$start" "$SLURM_JOB_ID")
    generated=false
    for attempt in 1 2 3; do
        temp_dir="${temp_base}_attempt_${attempt}"
        attempt_seed=$(( BASE_SEED + (attempt - 1) * 1000003 ))
        echo "generate start=$start frames=$frames attempt=$attempt temp=$temp_dir"
        if timeout 180s "$PYTHON" "$REPO_DIR/user_workflows/generate_course_shard.py" \
            --output-dir "$temp_dir" \
            --frames "$frames" \
            --start-index "$start" \
            --seed "$attempt_seed" \
            --width "$WIDTH" \
            --height "$HEIGHT" \
            --gate-texture-version "$GATE_TEXTURE_VERSION" \
            --camera-calibration "$REPO_DIR/gap8_perception/configs/hm01b0_calibration.json"; then
            if "$PYTHON" "$REPO_DIR/user_workflows/validate_course_shard.py" \
                --shard "$temp_dir" --expected "$frames"; then
                generated=true
                break
            fi
            echo "generation attempt $attempt produced invalid output" >&2
        else
            echo "generation attempt $attempt failed or timed out" >&2
        fi
    done
    if [[ "$generated" != true ]]; then
        echo "all generation attempts failed for start=$start" >&2
        exit 77
    fi

    "$PYTHON" "$REPO_DIR/user_workflows/compact_course_shard.py" \
        --shard "$temp_dir"

    touch "$temp_dir/_SUCCESS"
    mv "$temp_dir" "$final_dir"
    echo "completed $final_dir"
done
