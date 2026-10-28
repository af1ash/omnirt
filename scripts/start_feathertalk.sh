#!/usr/bin/env bash
set -euo pipefail
# Dataset and checkpoints can alternatively be provided by each WS init request.
export OMNIRT_FEATHERTALK_RUNTIME=1
exec "${OMNIRT_FEATHERTALK_PYTHON:-python}" -m uvicorn \
  omnirt.server.avatar_app:create_avatar_app --factory \
  --host "${OMNIRT_FEATHERTALK_HOST:-0.0.0.0}" \
  --port "${OMNIRT_FEATHERTALK_PORT:-9000}" "$@"
