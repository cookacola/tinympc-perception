#!/usr/bin/env bash
#SBATCH --job-name=gate_tex_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --exclude=a2r-main
#SBATCH --time=00:20:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/isaacsim-workspace
TEXTURE_VERSION=${TEXTURE_VERSION:-hm01b0_v2}
OUTPUT=$REPO/workspace/course_gate_texture_${TEXTURE_VERSION}_smoke
PARTIAL=${OUTPUT}.partial
PYTHON=/home/cchen/isaacsim-env/bin/python

cd "$REPO"
rm -rf "$PARTIAL"
"$PYTHON" user_workflows/generate_course_shard.py \
  --output-dir "$PARTIAL" --frames 100 --start-index 0 --seed 20260727 \
  --rt-subframes 8 \
  --gate-texture-version "$TEXTURE_VERSION" \
  --camera-calibration gap8_perception/configs/hm01b0_calibration.json
"$PYTHON" user_workflows/validate_course_shard.py \
  --shard "$PARTIAL" --expected 100
"$PYTHON" user_workflows/compact_course_shard.py --shard "$PARTIAL"
touch "$PARTIAL/_SUCCESS"
rm -rf "$OUTPUT"
mv "$PARTIAL" "$OUTPUT"
