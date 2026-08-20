#!/usr/bin/env bash
#SBATCH --job-name=joint-esp-dronet-gate
#SBATCH --nodelist=a2r-main
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

seeds=(1337 2027 3407)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
source_root=/home/cchen/tinympc-gate-texture
pulp_root=/home/cchen/pulp-dronet/tiny-pulp-dronet-v3
workspace=/home/cchen/isaacsim-workspace/workspace
output_root=${OUTPUT_ROOT:-$workspace/espnet_dronet_middle_gate_joint_v1}
output=$output_root/seed_${seed}

mkdir -p "$workspace/logs" "$output"
cd "$source_root"
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate pulp-dronet-v3-motion
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job: $SLURM_JOB_ID node: $(hostname) seed: $seed"
echo "Conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python user_workflows/train_joint_espnet_dronet_gate.py \
  --navigation-data "$pulp_root/dataset_original/Dataset_PULP_Dronet_v3" \
  --navigation-checkpoint "$workspace/espnet_dronet_navigation_official/seed_${seed}/best.pt" \
  --reference-manifest "$pulp_root/fair_comparison/seed_1337/sample_manifest.json" \
  --gate-checkpoint "$workspace/newbeedrone_joint_espnet_middle_finetune/best.pt" \
  --gate-dataset "$workspace/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$workspace/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$workspace/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$workspace/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$workspace/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --output "$output" --epochs "${EPOCHS:-30}" --patience "${PATIENCE:-7}" \
  --batch-size "${BATCH_SIZE:-96}" --gate-batch-size "${GATE_BATCH_SIZE:-48}" \
  --workers "$SLURM_CPUS_PER_TASK" --seed "$seed"
