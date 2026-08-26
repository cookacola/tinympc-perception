#!/usr/bin/env bash
#SBATCH --job-name=joint_dv3_gate
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=a2r-main
#SBATCH --time=06:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env

SOURCE=${SOURCE:-/home/cchen/tinympc-gate-texture}
WORKSPACE=${WORKSPACE:-/home/cchen/isaacsim-workspace/workspace}
PULP_ROOT=${PULP_ROOT:-/home/cchen/pulp-dronet/tiny-pulp-dronet-v3}
FROZEN=${FROZEN:-$WORKSPACE/newbeedrone_frozen_dronetv3_gate_taps_v1/block2/best.pt}
OUTPUT=${OUTPUT:-$WORKSPACE/newbeedrone_joint_dronetv3_block2_v1}

mkdir -p "$WORKSPACE/logs"
echo "Job: $SLURM_JOB_ID node: $(hostname)"
echo "Conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$SOURCE"

python user_workflows/train_joint_dronet_gate.py \
  --tap block2 \
  --checkpoint "$FROZEN" \
  --pulp-root "$PULP_ROOT" \
  --navigation-data "$PULP_ROOT/dataset_original/Dataset_PULP_Dronet_v3" \
  --baseline-result "$PULP_ROOT/fair_comparison/seed_2027/pulp_dronetv3_2frame_result.json" \
  --gate-dataset "$WORKSPACE/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$WORKSPACE/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --output "$OUTPUT" \
  --epochs "${EPOCHS:-15}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --limit "${LIMIT:-0}"
