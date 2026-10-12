#!/usr/bin/env bash
# Repeat the 441 runs of vits512_g112_kv28 and vits768_3layer (2026-10-12): their order in
# closed loop (768 > g112) is the reverse of the open-loop PDM order (g112 0.9533 > 768 0.9498).
# Same preparation, image, mounts and contract as the first runs; both at 5 drivers/renderers
# per GPU (40 workers) so the two repeats share one placement (the first runs used 40 / 48).
# Each waits for all eight GPUs to be free; a failed one does not stop the other.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TAG="${TAG:--rerun-20261012}"
log() { echo "[$(date '+%F %T')] $*"; }
for model in vits512_g112_kv28 vits768_3layer; do
    log "$model$TAG: queued"
    .venv/bin/python e2e_challenge/pkg3cam_eval/run_pkg3cam.py --model "$model" --tag="$TAG" --wait --prune-asl --replicas 5 \
        && log "$model$TAG: done" || log "$model$TAG: FAILED"
done
