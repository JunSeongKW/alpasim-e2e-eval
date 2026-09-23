#!/usr/bin/env bash
# axe-v9 with the route-corridor gate, on the full 441-scene curated_val split.
#
# The environment is held to run_merged_route_ep30.sh, which is how the same
# checkpoint was scored without the gate: dev preset, the same 441 scenes, one
# rollout each, 16 drivers and 16 renderers on GPUs 4-7, and gains lat 1.0 /
# lon 0.25 / idx 3. That run's summary is the baseline this one is read
# against, so every variable except the gate has to match it -- a difference
# anywhere else would land in the same number and be indistinguishable.
#
# No video: 441 clips of five-panel rendering is hours of matplotlib for
# footage nobody will watch. The ten-clip A/B runs exist for looking at.
#
# UV_OFFLINE: `uv run` revalidates direct-URL dependencies at launch, and a
# github.com outage stopped two runs that needed nothing from the network.
# Everything is already in the uv cache.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HERE="$ROOT/e2e_challenge/route_cache_filter"
AXE="$ROOT/e2e_challenge/axe_local_eval"
TAG="${TAG:-val441}"
LOG="$HERE/curatedval441_${TAG}.log"
RUN_DIR="$ROOT/runs/routecache-$TAG"
IMG=alpasim-e2e-drivesuprim-stage3:merged-route-ep30
# Replicas per GPU, lowered from the reference run's 4 because starting them is
# metadata-heavy: each controller is a `uv run` over a venv, and sixteen at once
# put every one of them into uninterruptible disk wait when another researcher's
# evaluation is already saturating the ext4 journal. Two failed attempts died
# exactly there. Fewer replicas start slower but they start, and on a shared
# disk yielding is cheaper than a run that never begins.
REPLICAS="${REPLICAS:-2}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-4,5,6,7}"
# Driver and renderer placement can be separated on a busy shared host. The
# driver model is relatively small while a renderer's active scene cache is
# large, so putting them on different cards leaves substantially more headroom
# than reducing the global worker count alone.
DRIVER_GPU_INDICES_CSV="${DRIVER_GPU_INDICES_CSV:-$GPU_INDICES_CSV}"
RENDER_GPU_INDICES_CSV="${RENDER_GPU_INDICES_CSV:-$GPU_INDICES_CSV}"
ROUTE_GATE_HORIZON_S="${ROUTE_GATE_HORIZON_S:-4.0}"
IFS=',' read -r -a driver_gpu_indices <<< "$DRIVER_GPU_INDICES_CSV"
N_DRIVER_GPUS="${#driver_gpu_indices[@]}"
N_DRIVERS="$((REPLICAS * N_DRIVER_GPUS))"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$N_DRIVERS}"
RENDERER_REPLICAS_PER_GPU="${RENDERER_REPLICAS_PER_GPU:-$REPLICAS}"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

rm -rf "$RUN_DIR"
log "=== val441 + route gate: horizon=${ROUTE_GATE_HORIZON_S}s, starting ${N_DRIVERS} drivers on GPUs ${DRIVER_GPU_INDICES_CSV}; renderers on ${RENDER_GPU_INDICES_CSV} ==="
addrs="$(IMAGE="$IMG" \
    CONTAINER_PREFIX="axe-rc441-$TAG" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV="$DRIVER_GPU_INDICES_CSV" REPLICAS_PER_GPU="$REPLICAS" \
    DRIVER_PKG="$HERE/drivesuprim_challenge" \
    MODEL_PY="$HERE/drivesuprim_model.py" \
    AGENT_PY="$HERE/drivesuprim_agent.py" \
    ROUTE_CACHE=1 ROUTE_CORRIDOR_M=4.0 \
    ROUTE_GATE_HORIZON_S="$ROUTE_GATE_HORIZON_S" \
    "$HERE/start_drivers_routecache.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }
log "drivers: $addrs"

log "=== 441 scenes x 1 rollout (dev), gains lat=1.0 lon=0.25 idx=3, route gate ON (${ROUTE_GATE_HORIZON_S}s), ${ROLLOUT_WORKERS} workers ==="
RUN_DIR="$RUN_DIR" \
UV_OFFLINE=1 \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS="$ROLLOUT_WORKERS" \
RENDER_GPUS_CSV="$RENDER_GPU_INDICES_CSV" \
RENDERER_REPLICAS_PER_GPU="$RENDERER_REPLICAS_PER_GPU" \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=true \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "val441: wizard exit=$?"

# run_curated_val.sh tries to drop the rollouts itself, but its cleanup is an
# EXIT trap and the script ends in `exec uv run alpasim_wizard` -- exec replaces
# the shell, so the trap never fires and a 441-scene run leaves 175 GB of
# rollout.asl behind. Nothing here reads them back: no video is rendered, and
# aggregate/results-summary.json is the whole product. Do the sweep here, and
# only once the summary exists, so a failed run still keeps its evidence.
if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    freed="$(du -sm "$RUN_DIR/rollouts" 2>/dev/null | cut -f1)"
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
    log "val441: removed rollouts/txt-logs/controller (~${freed:-0} MB)"
else
    log "val441: no summary, keeping rollouts for diagnosis"
fi

# The ROUTEGATE lines are the only per-frame record of what the gate did, and
# they go with the containers.
for c in $(docker ps -aq --filter "name=axe-rc441-$TAG"); do
    docker logs "$c" > "$RUN_DIR.driver-$c.log" 2>&1 || true
done
docker ps -aq --filter "name=axe-rc441-$TAG" | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    log "val441: OK -> $RUN_DIR/aggregate/results-summary.json"
else
    log "val441: FAILED, no summary"
    exit 1
fi
