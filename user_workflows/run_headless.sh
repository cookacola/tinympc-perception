#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_PY_ENV="${ISAACSIM_PY_ENV:-$HOME/isaacsim-env}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/workspace/datasets/demo}"

"$ISAACSIM_PY_ENV/bin/python" "$REPO_DIR/user_workflows/generate_dataset.py" \
  --headless --frames "${FRAMES:-100}" --output-dir "$OUTPUT_DIR"
"$ISAACSIM_PY_ENV/bin/python" "$REPO_DIR/user_workflows/inspect_annotations.py" \
  --dataset "$OUTPUT_DIR" --expected "${FRAMES:-100}"

