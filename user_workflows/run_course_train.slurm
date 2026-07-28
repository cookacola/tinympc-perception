#!/usr/bin/env bash
#SBATCH --job-name=hm01b0_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err

set -euo pipefail
REPO_DIR="/home/cchen/isaacsim-workspace"
DATASET="${DATASET:-$REPO_DIR/workspace/course_hm01b0_160_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/workspace/course_hm01b0_training}"

echo "job_id=$SLURM_JOB_ID node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
/home/cchen/isaacsim-env/bin/python "$REPO_DIR/user_workflows/train.py" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "${EPOCHS:-20}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
