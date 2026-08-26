#!/usr/bin/env bash
#SBATCH --job-name=confidence_matrix
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
ROOT=$WORKSPACE/newbeedrone_confidence_matrix_visible_v3
OBSTACLE_DATA=$WORKSPACE/tinympc_event_aligned_dataset_20260816
OBSTACLE_CKPT=$WORKSPACE/tinympc_event_aligned_finalists_20260816/espnetv2_lite_spatial_gate_finetune_seed270202/espnetv2_lite_spatial_gate_finetune.pt
GATE_INIT=$WORKSPACE/newbeedrone_joint_espnet_middle_finetune/best.pt
OLD_GATE=$WORKSPACE/newbeedrone_photo_varied_v2_corners_20k
OLD_TARGETS=$WORKSPACE/newbeedrone_photo_varied_v2_corner_targets
OLD_NEGATIVE=$WORKSPACE/newbeedrone_no_gate_v1_5k
OLD_MIXED=$WORKSPACE/newbeedrone_retained_obstacle_gate_v1/best.pt
NEW_GATE=$WORKSPACE/newbeedrone_multi_course_gate_v2_20k
NEW_TARGETS=$WORKSPACE/newbeedrone_multi_course_gate_v2_targets
NEW_NEGATIVE=$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k
NEW_MIXED=$WORKSPACE/newbeedrone_multi_course_retained_v2/best.pt

if [[ -e "$ROOT" ]]; then echo "refusing to overwrite $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
cd "$SOURCE"
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

common=(--obstacle-dataset "$OBSTACLE_DATA" --real-root /home/cchen/real_flight_data \
  --obstacle-checkpoint "$OBSTACLE_CKPT" --gate-checkpoint "$GATE_INIT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}")

run_eval() {
  local checkpoint=$1 gate=$2 targets=$3 negative=$4 output=$5
  python user_workflows/evaluate_gate_confidence_alternatives.py "${common[@]}" \
    --gate-dataset "$gate" --gate-targets "$targets" \
    --gate-split-file "$targets/split_shards.json" --no-gate-dataset "$negative" \
    --mixed-checkpoint "$checkpoint" --output "$ROOT/$output"
}
run_eval "$OLD_MIXED" "$OLD_GATE" "$OLD_TARGETS" "$OLD_NEGATIVE" old_on_old
run_eval "$OLD_MIXED" "$NEW_GATE" "$NEW_TARGETS" "$NEW_NEGATIVE" old_on_new
run_eval "$NEW_MIXED" "$OLD_GATE" "$OLD_TARGETS" "$OLD_NEGATIVE" new_on_old
run_eval "$NEW_MIXED" "$NEW_GATE" "$NEW_TARGETS" "$NEW_NEGATIVE" new_on_new
