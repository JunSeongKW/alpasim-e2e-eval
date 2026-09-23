#!/usr/bin/env bash
# Turn a finished rectification into a training-ready prepared root.
#
#     source setup_nurec.sh
#     bash scripts/nurec/finalize_data.sh
#
# Three steps, each of which fails loudly rather than producing a root that
# looks fine and is wrong:
#   1. rewrite the NAVSIM logs against the rectified images  (source untouched)
#   2. the preflight scripts/nurec/train*.sh runs before DDP
#   3. project the annotated boxes into the rectified images and report
set -euo pipefail

: "${AXE_ROOT:?source setup_nurec.sh first}"
: "${PYTHON_BIN:?source setup_nurec.sh first}"

rendered=$(find -L "$NUREC_RENDER_ROOT" -maxdepth 2 -name _complete 2>/dev/null | wc -l)
rectified=$(find -L "$NUREC_SENSOR_ROOT" -maxdepth 2 -name _complete 2>/dev/null | wc -l)
echo "rendered=$rendered  rectified=$rectified"
if [ "$rectified" -lt "$rendered" ]; then
  echo "  note: $((rendered-rectified)) clips rendered but not yet rectified;"
  echo "        this root will cover $rectified of them. Re-run after rectify_follow.sh catches up."
fi

echo
echo "== 1. rewriting logs =="
"$PYTHON_BIN" -m navsim.planning.data.nurec_rectify.logs \
  --source-prepared-root "$NUREC_SOURCE_ROOT" \
  --output-prepared-root "$NUREC_PREPARED_ROOT" \
  --calibration "$NUREC_CALIBRATION" \
  --image-root "$NUREC_SENSOR_ROOT" | tail -12

echo
echo "== 2. preflight =="
"$PYTHON_BIN" -m navsim.planning.data.nurec_validate \
  --log-root "$NUREC_PREPARED_ROOT/navsim_logs/trainval" \
  --sensor-root "$NUREC_SENSOR_ROOT" \
  --map-root "$NUREC_MAP_ROOT" | tail -10

echo
echo "== 3. box projection =="
"$PYTHON_BIN" "$AXE_ROOT/scripts/nurec/verify_rectified_projection.py" \
  --log-root "$NUREC_PREPARED_ROOT/navsim_logs/trainval" \
  --image-root "$NUREC_SENSOR_ROOT" \
  --out-dir "$NUREC_VALIDATION_ROOT/verify_$(basename "$(readlink -f "$NUREC_SENSOR_ROOT")")" \
  --limit 20 --overlay-logs 4

echo
echo "== target K, training vs driver =="
"$PYTHON_BIN" -m pytest "$AXE_ROOT/tests/test_nurec_rectified_geometry.py" -q -p no:warnings 2>&1 | tail -3
