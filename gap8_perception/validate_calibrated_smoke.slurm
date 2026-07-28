#!/usr/bin/env bash
#SBATCH --job-name=camera_geometry
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail
cd /home/cchen/isaacsim-workspace
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
  gap8_perception/validate_camera_geometry.py \
  workspace/course_hm01b0_calibrated_smoke/shard_000000000 \
  --calibration gap8_perception/configs/hm01b0_calibration.json \
  --output workspace/course_hm01b0_calibrated_smoke/camera_geometry_report.json
