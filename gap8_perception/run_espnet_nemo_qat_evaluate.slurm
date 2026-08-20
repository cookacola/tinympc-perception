#!/usr/bin/env bash
#SBATCH --job-name=espnet_qat_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=a2r-main
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

export PYTHONHASHSEED=${NEMO_GRAPH_HASH_SEED:-9}
REPO=/home/cchen/gap8-drone-racing-perception
TRAINING_REPO=/home/cchen/tinympc-gate-texture
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
DATASET=$WORKSPACE/tinympc_event_aligned_dataset_20260816
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_qat_deployment}
PTQ_OUTPUT=$WORKSPACE/espnet_dory_student_v2_deployment
QAT=${QAT:-$WORKSPACE/espnet_dory_student_v2_qat}
GATE_DATASET=$WORKSPACE/newbeedrone_multi_course_gate_v2_20k
GATE_TARGETS=$WORKSPACE/newbeedrone_multi_course_gate_v2_targets
NO_GATE=$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k
PAIRED_NO_GATE=$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k
TEACHER=$WORKSPACE/newbeedrone_multi_course_safety_constrained_v3/selected_for_deployment.pt
OBSTACLE_CHECKPOINT=$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt
GATE_CHECKPOINT=$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt

mkdir -p "$OUTPUT"
cd "$REPO"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /home/cchen/miniconda3/envs/frontnet/bin/python \
  gap8_perception/nemo_espnet_dory_student_export.py \
  --bridge "$PTQ_OUTPUT/bridge" --dataset "$DATASET" \
  --qat-directory "$QAT" --output "$OUTPUT/integer"

cd "$TRAINING_REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/evaluate_espnet_dory_onnx.py \
  --integer-dir "$OUTPUT/integer" \
  --nemo-report "$OUTPUT/integer/nemo_espnet_dory_student_report.json" \
  --obstacle-dataset "$DATASET" \
  --gate-dataset "$GATE_DATASET" --gate-targets "$GATE_TARGETS" \
  --gate-split-file "$GATE_TARGETS/split_shards.json" \
  --no-gate-dataset "$NO_GATE" --paired-no-gate-dataset "$PAIRED_NO_GATE" \
  --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$OBSTACLE_CHECKPOINT" \
  --gate-checkpoint "$GATE_CHECKPOINT" --teacher-checkpoint "$TEACHER" \
  --output "$OUTPUT/integer_ground_truth.json" \
  --maximum-per-dataset "${EVAL_MAX:-0}"

echo "QAT integer evaluation completed at: $(date)"
