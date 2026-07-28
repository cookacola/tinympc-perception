#!/usr/bin/env bash
#SBATCH --job-name=verify_overfit
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
: "${OVERFIT_LOG:?set OVERFIT_LOG}"
/home/cchen/isaacsim-env/bin/python gap8_perception/verify_overfit.py "$OVERFIT_LOG"
