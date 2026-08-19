#!/usr/bin/env bash
#SBATCH --job-name=espnet_nemo_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=a2r-main
#SBATCH --time=12:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
BRIDGE=${BRIDGE:-$WORKSPACE/espnet_dory_student_v2_deployment/bridge}
DATASET=$WORKSPACE/tinympc_event_aligned_dataset_20260816
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_qat}
mkdir -p "$OUTPUT"
cd "$REPO"
for graph in encoder corner_head gate_head danger_head; do
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
    /home/cchen/miniconda3/envs/frontnet/bin/python \
    gap8_perception/train_nemo_espnet_dory_qat.py \
    --graph "$graph" --bridge "$BRIDGE" --dataset "$DATASET" \
    --output "$OUTPUT" --epochs "${EPOCHS:-4}" \
    --train-limit "${TRAIN_LIMIT:-4096}" --batch-size 32 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --learning-rate 1e-5
done
