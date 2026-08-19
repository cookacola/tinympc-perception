#!/usr/bin/env bash
#SBATCH --job-name=safe_gate_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
OUTPUT=${OUTPUT:-$WORKSPACE/newbeedrone_multi_course_safety_constrained_v3}
mkdir -p "$WORKSPACE/logs"
cd "$SOURCE"
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python user_workflows/train_retained_obstacle_gate.py \
  --obstacle-dataset "$WORKSPACE/tinympc_event_aligned_dataset_20260816" \
  --gate-dataset "$WORKSPACE/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt" \
  --gate-checkpoint "$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt" \
  --initial-mixed-checkpoint "$WORKSPACE/newbeedrone_multi_course_retained_v2/best.pt" \
  --output "$OUTPUT" --epochs 15 --save-every-epoch \
  --encoder-learning-rate 1e-5 --obstacle-learning-rate 1e-4 \
  --gate-learning-rate 3e-4 --obstacle-weight 8.0 \
  --minimum-unsafe-recall 0.963 --minimum-collision-ap 0.90 \
  --maximum-real-corner-error 20.0 --maximum-no-gate-fpr 0.05 \
  --obstacle-batch-size 32 --gate-batch-size 64 \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
