#!/usr/bin/env bash
#SBATCH --job-name=safe_gate_conf
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=/home/cchen/tinympc-gate-texture
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
CHECKPOINT=${CHECKPOINT:-$WORKSPACE/newbeedrone_multi_course_safety_constrained_v3/epoch_011.pt}
OUTPUT=${OUTPUT:-$WORKSPACE/newbeedrone_safety_epoch11_confidence_v1}
cd "$SOURCE"
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX checkpoint: $CHECKPOINT"
python user_workflows/evaluate_gate_confidence_alternatives.py \
  --obstacle-dataset "$WORKSPACE/tinympc_event_aligned_dataset_20260816" \
  --gate-dataset "$WORKSPACE/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt" \
  --gate-checkpoint "$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt" \
  --mixed-checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
