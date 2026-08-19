#!/usr/bin/env bash
#SBATCH --job-name=course_preview
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env

SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
OUTPUT=${OUTPUT:-$WORKSPACE/newbeedrone_multi_course_preview_v1}
TEXTURES=${TEXTURES:-$WORKSPACE/commons_clutter_textures_v1/manifest.json}
FRAMES=${FRAMES:-7}

echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
mkdir -p "$OUTPUT" "$WORKSPACE/logs"
cd "$SOURCE"

for family_index in 0 1 2 3 4 5; do
  start=$(( family_index * FRAMES ))
  for mode in gate no_gate; do
    destination=$(printf '%s/train_family_%02d_%s' "$OUTPUT" "$family_index" "$mode")
    if [[ -e "$destination" ]]; then
      echo "refusing to overwrite preview: $destination" >&2
      exit 2
    fi
    extra=()
    if [[ "$mode" == no_gate ]]; then extra+=(--no-gates --gate-target-probability 0); fi
    completed=false
    for attempt in 1 2 3; do
      partial="${destination}.job_${SLURM_JOB_ID}.attempt_${attempt}"
      if [[ -e "$partial" ]]; then find "$partial" -depth -delete; fi
      if timeout --signal=TERM --kill-after=20s 120s \
        python user_workflows/generate_course_shard.py \
          --output-dir "$partial" --frames "$FRAMES" --start-index "$start" \
          --seed 20260819 --width 160 --height 160 --rt-subframes 1 \
          --gate-texture-version photo_v1 --gate-target-probability 0.85 \
          --layout-family multi_course_v2 --asset-split train \
          --clutter-texture-manifest "$TEXTURES" \
          --camera-calibration gap8_perception/configs/hm01b0_calibration.json \
          "${extra[@]}"; then
        validation=(--shard "$partial" --expected "$FRAMES")
        if [[ "$mode" == no_gate ]]; then validation+=(--allow-no-gates); fi
        if python user_workflows/validate_course_shard.py "${validation[@]}"; then
          mv "$partial" "$destination"
          completed=true
          break
        fi
      fi
      echo "preview attempt $attempt failed for family=$family_index mode=$mode" >&2
      if [[ -e "$partial" ]]; then find "$partial" -depth -delete; fi
    done
    if [[ "$completed" != true ]]; then
      echo "all preview attempts failed for family=$family_index mode=$mode" >&2
      exit 77
    fi
  done
done

echo "Finished: $(date)"
