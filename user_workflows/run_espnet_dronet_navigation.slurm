#!/usr/bin/env bash
#SBATCH --job-name=espnet-dronet-head
#SBATCH --nodelist=a2r-main
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%A_%a.err
set -euo pipefail

seeds=(1337 2027 3407)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
source_root=/home/cchen/tinympc-gate-texture
pulp_root=/home/cchen/pulp-dronet/tiny-pulp-dronet-v3
workspace=/home/cchen/isaacsim-workspace/workspace
output=$workspace/espnet_dronet_navigation_official/seed_${seed}

mkdir -p "$workspace/logs" "$output"
cd "$source_root"
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate pulp-dronet-v3-motion
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job: $SLURM_JOB_ID node: $(hostname) seed: $seed"
echo "Conda: $CONDA_PREFIX python: $(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python user_workflows/train_espnet_dronet_navigation.py \
  --data "$pulp_root/dataset_original/Dataset_PULP_Dronet_v3" \
  --base-checkpoint "$workspace/espnet_dory_student_v2/best.pt" \
  --reference-manifest "$pulp_root/fair_comparison/seed_1337/sample_manifest.json" \
  --output "$output" \
  --epochs 50 \
  --patience 8 \
  --batch-size 256 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --seed "$seed"
