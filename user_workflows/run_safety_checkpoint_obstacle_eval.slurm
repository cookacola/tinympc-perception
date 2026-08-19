#!/usr/bin/env bash
#SBATCH --job-name=safe_obstacle_eval
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
SOURCE=/home/cchen/tinympc-gate-texture
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
RUN=$WORKSPACE/newbeedrone_multi_course_safety_constrained_v3
cd "$SOURCE"
python user_workflows/evaluate_retained_obstacle_checkpoints.py \
  --obstacle-dataset "$WORKSPACE/tinympc_event_aligned_dataset_20260816" \
  --obstacle-checkpoint "$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt" \
  --gate-checkpoint "$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt" \
  --checkpoints "$RUN/epoch_001.pt" "$RUN/epoch_011.pt" "$RUN/epoch_015.pt" \
  --output "$RUN/candidate_obstacle_evaluation.json" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
