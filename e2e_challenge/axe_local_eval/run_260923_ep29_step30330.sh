#!/usr/bin/env bash
# Evaluate the 2026-09-23 epoch-29 checkpoint under the exact AXE-v9 local
# contract: AXE-v9 runtime image, dev preset, 441 curated-val scenes, one
# rollout, 16 contestant replicas, 16 workers/renderers, and tuned v9 MPC gains.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
RUN_DIR="$ROOT/runs/leaderboard-260923-ep29-step30330"
LOG="$AXE/260923_ep29_step30330.log"
IMG=alpasim-e2e-axe-v9:260923-ep29-step30330
SHA=7e0e59be63146c72e917989d30ce778ce807e7971386d61c66e5d6f925db289b
PREFIX=axe-lb-260923-e29s30330

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

[[ ! -e "$RUN_DIR" ]] || { log "refusing to overwrite $RUN_DIR"; exit 2; }
actual_sha="$(docker image inspect "$IMG" \
    --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_sha" == "$SHA" ]] || {
    log "checkpoint image mismatch: expected=$SHA actual=${actual_sha:-missing}"
    exit 2
}

# Which cards to use. The default is the four this account is normally given,
# but the assignment moves: on 2026-09-23 another researcher's paired
# evaluation held 4-7 for eight hours while 0-3 sat empty, and the user
# reassigned this run there. Nothing else about the contract changes -- still
# four cards, still sixteen replicas spread across them.
GPUS="${GPUS:-4,5,6,7}"

# Loading sixteen checkpoints onto the cards takes ten minutes or more on a
# busy disk, and the drivers are separate containers the wizard only talks to
# over localhost -- so when a retry is about the simulator side (a mount, a
# probe timeout), the drivers from the previous attempt can be kept. Set
# REUSE_DRIVERS=1 and every "$PREFIX-g*" container that is up and reports
# "policy load complete" is used as-is; if fewer than sixteen qualify the
# script falls back to starting a fresh set, so a stale or half-dead pool can
# never be silently used.
reuse_ok=0
if [[ "${REUSE_DRIVERS:-0}" == 1 ]]; then
    mapfile -t live < <(docker ps --filter "name=^${PREFIX}-g" --format '{{.Names}}' | sort)
    ready=0
    for c in "${live[@]}"; do
        docker logs "$c" 2>&1 | grep -q "policy load complete" && ready=$((ready + 1))
    done
    if [[ ${#live[@]} -eq 16 && $ready -eq 16 ]]; then
        # Ports are BASE_PORT + index, the same order start_drivers.sh uses.
        addrs="[$(for i in $(seq 0 15); do printf '"localhost:%d"' $((6900 + i)); [[ $i -lt 15 ]] && printf ','; done)]"
        reuse_ok=1
        log "reusing 16 live driver replicas: $addrs"
    else
        log "REUSE_DRIVERS=1 but only ${#live[@]} up / $ready ready; starting fresh"
    fi
fi

# Sixteen-way AXE-v9 placement needs all four cards.  Do not collide with
# another researcher's native training job: wait until each card is effectively
# empty, then recheck immediately before launching the replicas.
# Skipped when the previous attempt's drivers are being reused: they are what
# occupies the cards, and waiting for them to vacate would never return.
while [[ $reuse_ok -eq 0 ]]; do
    mapfile -t used < <(nvidia-smi -i "$GPUS" \
        --query-gpu=memory.used --format=csv,noheader,nounits)
    ready=1
    for mib in "${used[@]}"; do
        (( mib < 2000 )) || ready=0
    done
    (( ready == 1 )) && break
    log "waiting for GPUs $GPUS; used MiB: ${used[*]}"
    sleep 300
done

if [[ $reuse_ok -eq 0 ]]; then
log "GPUs $GPUS available; starting 16 AXE-v9 driver replicas"
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256="$SHA" \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="$PREFIX" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }
log "drivers ready: $addrs"
fi

cat > "$RUN_DIR.provenance.json" <<EOF
{
  "checkpoint": "/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/models/260923_epoch=29-step=30330.ckpt",
  "checkpoint_sha256": "$SHA",
  "image": "$IMG",
  "base_runtime": "696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9",
  "preset": "dev",
  "scenes": 441,
  "rollouts_per_scene": 1,
  "gpus": [4, 5, 6, 7],
  "drivers": 16,
  "workers": 16,
  "renderer_replicas_per_gpu": 4,
  "mpc": {"long_position_weight": 0.25, "lat_position_weight": 1.0, "idx_start_penalty": 3}
}
EOF

log "starting 441 scenes x 1 rollout; 16 workers; no video"
# Startup speed (renderer shared cache, PYTHONDONTWRITEBYTECODE, UV_OFFLINE)
# is handled inside run_curated_val.sh now (FAST_STARTUP=1 default). Four
# attempts at this evaluation died in startup: a 1792 s renderer probe timeout
# while sixteen renderers each downloaded 2.8 GB; a cache mounted at a path
# the image never reads; and a .pyc write storm from 37 fresh python processes
# with nothing bound eight minutes in. All of that is placement and caching --
# the 16 drivers / 16 workers / 16 renderers / 441 scenes contract is untouched.
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=16 \
RENDER_GPUS_CSV="$GPUS" \
RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=true \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=false \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
status=$?
log "wizard exit=$status"

if [[ -f "$RUN_DIR/docker-compose.yaml" ]]; then
    docker compose -f "$RUN_DIR/docker-compose.yaml" down \
        --timeout 10 --remove-orphans >/dev/null 2>&1 || true
fi
docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    freed="$(du -sm "$RUN_DIR/rollouts" 2>/dev/null | cut -f1)"
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
    mv "$RUN_DIR.provenance.json" "$RUN_DIR/evaluation_provenance.json"
    log "OK: $RUN_DIR/aggregate/results-summary.json; removed raw rollout data (~${freed:-0} MB)"
else
    log "FAILED: no summary; preserving run evidence"
    exit "${status:-1}"
fi
