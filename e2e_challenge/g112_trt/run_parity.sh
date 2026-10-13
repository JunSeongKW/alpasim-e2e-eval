#!/usr/bin/env bash
# 441 closed-loop run of the TensorRT image as built, compared with the two
# PyTorch 441 runs of vits512_g112_kv28.
#
#   e2e_challenge/g112_trt/run_parity.sh [IMAGE]
#
# The image runs exactly as it would be submitted: nothing mounted, no
# DRIVESUPRIM_* passed (run_pkg3cam model vits512_g112_kv28_trt, "baked").
# Everything else is the g112 runs' contract: dev preset, 441 curated_val, one
# rollout per clip, gains lat 1.0 / lon 0.25 / idx 3, batch 1. Only on this
# account's GPUs 4-7 (6 drivers + 6 renderers each, 24 workers), which changes
# throughput, not what is scored. rollout.asl is pruned once scored (disk).
# Then: every driver log must show the exact engine active and no fallback,
# and compare_parity.py must find the run inside PyTorch run-to-run noise.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
IMAGE="${1:-alpasim-e2e-drivesuprim-stage3:g112-kv28-trt}"
PREP=runs/prepare-stage3-vits512-g112-kv28-trt-20261013
RUN=leaderboard-stage3-vits512-g112-kv28-trt-20261013
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "$(python3 "$HERE/register_trt_model.py" "$IMAGE")"

PY=.venv/bin/python
R=e2e_challenge/pkg3cam_eval/run_pkg3cam.py
log "smoke (GPU 4)"
$PY $R --model vits512_g112_kv28_trt --smoke 4
grep -q "TRT backbone: active engine=exact" "$PREP/smoke/smoke.log" || { log "smoke: engine not active"; exit 1; }
log "441 run on GPUs 4-7"
$PY $R --model vits512_g112_kv28_trt --gpus 4,5,6,7 --replicas 6 --prune-asl --wait
# clips lost to a renderer/RPC error (not the model) run again, still on 4-5
$PY e2e_challenge/pkg3cam_eval/retry_infra.py --model vits512_g112_kv28_trt --gpus 4,5

L="runs/$RUN/driver-logs"
n=$(ls "$L"/*.log | wc -l)
active=$(grep -l "TRT backbone: active engine=exact" "$L"/*.log | wc -l)
fallback=$(cat "$L"/*.log | grep -cE "TRT backbone: call [0-9]+" || true)
log "driver logs: $n, exact engine active in $active, fallback/shape-miss lines $fallback"
(( active == n && fallback == 0 )) || { log "TRT not active everywhere"; exit 1; }
$PY "$HERE/compare_parity.py"
