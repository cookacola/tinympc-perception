#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 USER@LOCAL_HOST:/destination/" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -avP --prune-empty-dirs \
  --include='*/' \
  --include='hm01b0_segmenter_best.pt' \
  --include='policy_only_*.pt' \
  --include='log.csv' \
  --include='camera_sensor.json' \
  --include='inspection_report.json' \
  --include='inspection_contact_sheet.jpg' \
  --include='rgb_0000.png' \
  --include='hm01b0_mono_0000.png' \
  --include='depth_mm_0000.png' \
  --include='semantic_segmentation_0000.png' \
  --include='semantic_segmentation_labels_0000.json' \
  --include='prediction_*.png' \
  --include='inspection/overlay_0000.jpg' \
  --include='inspection/overlay_0025.jpg' \
  --include='inspection/overlay_0050.jpg' \
  --include='inspection/overlay_0075.jpg' \
  --include='inspection/overlay_0099.jpg' \
  --exclude='*' \
  "$REPO_DIR/workspace/" "$1"
