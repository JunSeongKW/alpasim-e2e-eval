#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${NUREC_MANIFEST:?Set NUREC_MANIFEST to the JSONL export}"
: "${NUREC_SENSOR_ROOT:?Set NUREC_SENSOR_ROOT to the mounted sensor directory}"
: "${NUREC_PREPARED_ROOT:?Set NUREC_PREPARED_ROOT to an output directory}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="$(command -v python || command -v python3)"
"$PYTHON_BIN" -m navsim.planning.data.nurec_converter \
  --manifest "$NUREC_MANIFEST" --sensor-root "$NUREC_SENSOR_ROOT" \
  --output-root "$NUREC_PREPARED_ROOT" "$@"
