#!/usr/bin/env bash
#SBATCH --job-name=esp-nav-calibrate
#SBATCH --nodelist=a2r-main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
repo=/home/cchen/gap8-drone-racing-perception
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
cd "$repo"
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"
PYTHONPATH=. python gap8_perception/calibrate_integer_navigation_outputs.py \
  --integer-dir "${INTEGER_DIR:?set INTEGER_DIR}" --report "${REPORT:?set REPORT}" \
  --navigation-data /home/cchen/pulp-dronet/tiny-pulp-dronet-v3/dataset_original/Dataset_PULP_Dronet_v3 \
  --output "${OUTPUT:?set OUTPUT}"
