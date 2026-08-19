#!/usr/bin/env bash
#SBATCH --job-name=no_gate_sdg
#SBATCH --array=0-4%1
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
SOURCE_DIR=${SOURCE_DIR:-/home/cchen/tinympc-gate-texture}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_no_gate_v1_5k}
FRAMES_PER_SHARD=${FRAMES_PER_SHARD:-1000}
start=$((SLURM_ARRAY_TASK_ID * FRAMES_PER_SHARD))
final_dir=$(printf "%s/shard_%09d" "$OUTPUT_ROOT" "$start")
partial_dir=$(printf "%s/.partial_%09d_%s" "$OUTPUT_ROOT" "$start" "$SLURM_JOB_ID")
mkdir -p "$OUTPUT_ROOT" /home/cchen/isaacsim-workspace/workspace/logs
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"
nvidia-smi
cd "$SOURCE_DIR"
python user_workflows/generate_course_shard.py --output-dir "$partial_dir" --frames "$FRAMES_PER_SHARD" --start-index "$start" --seed 20260819 --width 160 --height 160 --gate-texture-version photo_v1 --gate-target-probability 0 --no-gates --camera-calibration gap8_perception/configs/hm01b0_calibration.json
python user_workflows/validate_course_shard.py --shard "$partial_dir" --expected "$FRAMES_PER_SHARD" --allow-no-gates
python user_workflows/compact_course_shard.py --shard "$partial_dir"
touch "$partial_dir/_SUCCESS"
mv "$partial_dir" "$final_dir"
