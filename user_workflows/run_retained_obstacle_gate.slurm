#!/usr/bin/env bash
#SBATCH --job-name=retain_obs_gate
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=08:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}; WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$SOURCE"
python user_workflows/train_retained_obstacle_gate.py \
  --obstacle-dataset "$WORKSPACE/tinympc_event_aligned_dataset_20260816" \
  --gate-dataset "$WORKSPACE/newbeedrone_photo_varied_v2_corners_20k" \
  --gate-targets "$WORKSPACE/newbeedrone_photo_varied_v2_corner_targets" \
  --gate-split-file "$WORKSPACE/newbeedrone_photo_varied_v2_corner_targets/split_shards.json" \
  --no-gate-dataset "$WORKSPACE/newbeedrone_no_gate_v1_5k" \
  --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt" \
  --gate-checkpoint "$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt" \
  --output "$WORKSPACE/newbeedrone_retained_obstacle_gate_v1" \
  --epochs "${EPOCHS:-20}" --workers "${SLURM_CPUS_PER_TASK:-8}"
