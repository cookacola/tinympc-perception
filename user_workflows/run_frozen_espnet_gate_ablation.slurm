#!/usr/bin/env bash
#SBATCH --job-name=frozen_esp_gate
#SBATCH --array=0-5%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE_DIR=${SOURCE_DIR:-/home/cchen/tinympc-gate-texture}
DATASET=${DATASET:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_photo_v1_corners_20k}
TARGETS=${TARGETS:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_photo_v1_corner_targets}
CHECKPOINT=${CHECKPOINT:-/home/cchen/isaacsim-workspace/workspace/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_frozen_espnet_gate_ablation}
taps=(early middle late early middle late); representations=(binary binary binary sdf sdf sdf)
tap=${taps[$SLURM_ARRAY_TASK_ID]}; representation=${representations[$SLURM_ARRAY_TASK_ID]}
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$SOURCE_DIR"
python -m gap8_perception.train_frozen_espnet_head --task gate --tap "$tap" --representation "$representation" --checkpoint "$CHECKPOINT" --dataset "$DATASET" --targets "$TARGETS" --split-file "$TARGETS/split_shards.json" --output "$OUTPUT_ROOT/${tap}_${representation}" --epochs "${EPOCHS:-60}" --batch-size "${BATCH_SIZE:-128}" --workers "${SLURM_CPUS_PER_TASK:-8}"
