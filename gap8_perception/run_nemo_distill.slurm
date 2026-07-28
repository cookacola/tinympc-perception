#!/usr/bin/env bash
#SBATCH --job-name=gap8_nemo_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${BRIDGE:?set BRIDGE}"
: "${OUTPUT:?set OUTPUT}"
DATASET=${DATASET:-workspace/course_hm01b0_nbd_texture_50k}
EPOCHS=${EPOCHS:-12}
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH=/home/cchen/isaacsim-env/lib/python3.12/site-packages:/home/cchen/miniconda3/envs/frontnet/lib/python3.7/site-packages:. \
/home/cchen/isaacsim-env/bin/python \
  gap8_perception/train_nemo_distill.py \
  --bridge "$BRIDGE" --dataset "$DATASET" --output "$OUTPUT" \
  --epochs "$EPOCHS" --batch-size 32 --workers 8 \
  "${RESUME_ARGS[@]}"
