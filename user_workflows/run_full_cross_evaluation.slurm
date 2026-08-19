#!/usr/bin/env bash
#SBATCH --job-name=gate_cross_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
ROOT=$WORKSPACE/newbeedrone_full_cross_evaluation_v2
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

python user_workflows/evaluate_mixed_checkpoint_cross_dataset.py "${common[@]}" \
  --gate-dataset "$NEW_GATE" --gate-targets "$NEW_TARGETS" \
  --gate-split-file "$NEW_TARGETS/split_shards.json" --no-gate-dataset "$NEW_NEGATIVE" \
  --mixed-checkpoint "$OLD_MIXED" --output "$ROOT/old_checkpoint_on_new_test.json"
python user_workflows/evaluate_mixed_checkpoint_cross_dataset.py "${common[@]}" \
  --gate-dataset "$OLD_GATE" --gate-targets "$OLD_TARGETS" \
  --gate-split-file "$OLD_TARGETS/split_shards.json" --no-gate-dataset "$OLD_NEGATIVE" \
  --mixed-checkpoint "$NEW_MIXED" --output "$ROOT/new_checkpoint_on_old_test.json"

python user_workflows/evaluate_gate_confidence_alternatives.py "${common[@]}" \
  --gate-dataset "$NEW_GATE" --gate-targets "$NEW_TARGETS" \
  --gate-split-file "$NEW_TARGETS/split_shards.json" --no-gate-dataset "$NEW_NEGATIVE" \
  --mixed-checkpoint "$OLD_MIXED" --output "$ROOT/old_confidence_on_new_domain"
python user_workflows/evaluate_gate_confidence_alternatives.py "${common[@]}" \
  --gate-dataset "$OLD_GATE" --gate-targets "$OLD_TARGETS" \
  --gate-split-file "$OLD_TARGETS/split_shards.json" --no-gate-dataset "$OLD_NEGATIVE" \
  --mixed-checkpoint "$NEW_MIXED" --output "$ROOT/new_confidence_on_old_domain"
