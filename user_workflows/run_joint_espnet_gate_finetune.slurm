#!/usr/bin/env bash
#SBATCH --job-name=joint_esp_finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE_DIR=${SOURCE_DIR:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
DATASET=${DATASET:-$WORKSPACE/newbeedrone_photo_v1_corners_20k}
TARGETS=${TARGETS:-$WORKSPACE/newbeedrone_photo_v1_corner_targets}
BACKBONE=${BACKBONE:-$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt}
CORNER=${CORNER:-$WORKSPACE/newbeedrone_frozen_espnet_corner_ablation/middle_heatmap/best.pt}
GATE=${GATE:-$WORKSPACE/newbeedrone_frozen_espnet_gate_ablation/middle_binary/best.pt}
OUTPUT=${OUTPUT:-$WORKSPACE/newbeedrone_joint_espnet_middle_finetune}
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$SOURCE_DIR"
python -m gap8_perception.train_joint_espnet_gate --backbone-checkpoint "$BACKBONE" --corner-checkpoint "$CORNER" --gate-checkpoint "$GATE" --dataset "$DATASET" --targets "$TARGETS" --split-file "$TARGETS/split_shards.json" --output "$OUTPUT" --epochs "${EPOCHS:-60}" --batch-size "${BATCH_SIZE:-128}" --workers "${SLURM_CPUS_PER_TASK:-8}"
