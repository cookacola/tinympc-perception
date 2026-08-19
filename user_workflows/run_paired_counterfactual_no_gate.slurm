#!/usr/bin/env bash
#SBATCH --job-name=paired_nogate_sdg
#SBATCH --array=0-19%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
GATE_ROOT=${GATE_ROOT:-$WORKSPACE/newbeedrone_multi_course_gate_v2_20k}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k}
TEXTURES=${TEXTURES:-$WORKSPACE/commons_clutter_textures_v1/manifest.json}
FRAMES_PER_SHARD=${FRAMES_PER_SHARD:-1000}
start=$((SLURM_ARRAY_TASK_ID * FRAMES_PER_SHARD))
source_shard=$(printf "%s/shard_%09d" "$GATE_ROOT" "$start")
final_dir=$(printf "%s/shard_%09d" "$OUTPUT_ROOT" "$start")
if (( SLURM_ARRAY_TASK_ID < 16 )); then asset_split=train
elif (( SLURM_ARRAY_TASK_ID < 18 )); then asset_split=validation
else asset_split=test
fi

mkdir -p "$OUTPUT_ROOT" "$WORKSPACE/logs"
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID node: $(hostname) split: $asset_split conda: $CONDA_PREFIX python: $(which python)"
nvidia-smi
cd "$SOURCE"
if [[ -f "$final_dir/_SUCCESS" ]]; then echo "skip completed $final_dir"; exit 0; fi
if [[ ! -f "$source_shard/_SUCCESS" ]]; then echo "missing paired source $source_shard" >&2; exit 3; fi
source_seed=$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p["layout_seed"])-int(sys.argv[2]))' "$source_shard/scene_metadata.json" "$start")

generated=false
for attempt in 1 2 3 4 5; do
  partial=$(printf "%s/.partial_%09d_%s_attempt_%s" "$OUTPUT_ROOT" "$start" "$SLURM_JOB_ID" "$attempt")
  if [[ -e "$partial" ]]; then find "$partial" -depth -delete; fi
  if timeout --signal=TERM --kill-after=20s 300s \
    python user_workflows/generate_course_shard.py \
      --output-dir "$partial" --frames "$FRAMES_PER_SHARD" --start-index "$start" \
      --seed "$source_seed" --width 160 --height 160 --rt-subframes 1 \
      --lighting-scale 3.0 --gate-texture-version photo_v1 \
      --gate-target-probability 0.85 --no-gates --counterfactual-gate-camera-policy \
      --layout-family multi_course_v2 --asset-split "$asset_split" \
      --clutter-texture-manifest "$TEXTURES" \
      --camera-calibration gap8_perception/configs/hm01b0_calibration.json \
    && python user_workflows/validate_course_shard.py \
      --shard "$partial" --expected "$FRAMES_PER_SHARD" --allow-no-gates; then
    generated=true
    break
  fi
  echo "attempt $attempt failed for paired shard $start seed=$source_seed" >&2
  if [[ -e "$partial" ]]; then find "$partial" -depth -delete; fi
  sleep 15
done
if [[ "$generated" != true ]]; then echo "all attempts failed for shard $start" >&2; exit 77; fi
python user_workflows/compact_course_shard.py --shard "$partial"
touch "$partial/_SUCCESS"
mv "$partial" "$final_dir"
