#!/usr/bin/env bash
#SBATCH --job-name=frozen_dv3_gate
#SBATCH --array=0-2%1
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

SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
PULP_ROOT=${PULP_ROOT:-/home/cchen/pulp-dronet/tiny-pulp-dronet-v3}
CHECKPOINT=${CHECKPOINT:-$PULP_ROOT/fair_comparison/seed_2027/pulp_dronetv3_2frame_best.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/newbeedrone_frozen_dronetv3_gate_taps_v1}
taps=(block1 block2 block3)
tap=${taps[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$WORKSPACE/logs"
echo "Job: $SLURM_JOB_ID task: $SLURM_ARRAY_TASK_ID node: $(hostname)"
echo "Conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$SOURCE"

python user_workflows/train_frozen_dronet_gate_heads.py \
  --tap "$tap" \
  --checkpoint "$CHECKPOINT" \
  --pulp-root "$PULP_ROOT" \
  --gate-dataset "$WORKSPACE/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --output "$OUTPUT_ROOT/$tap" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --limit "${LIMIT:-0}"
