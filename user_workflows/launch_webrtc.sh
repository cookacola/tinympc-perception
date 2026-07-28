#!/usr/bin/env bash
set -euo pipefail

ISAACSIM_PY_ENV="${ISAACSIM_PY_ENV:-$HOME/isaacsim-env}"
ISAACSIM_HOST="${ISAACSIM_HOST:-$(hostname -I | awk '{print $1}')}"
SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49100}"
STREAM_PORT="${ISAACSIM_STREAM_PORT:-47998}"

exec "$ISAACSIM_PY_ENV/bin/isaacsim" isaacsim.exp.full.streaming \
  --no-window \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp="$ISAACSIM_HOST" \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort="$SIGNAL_PORT" \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort="$STREAM_PORT"

