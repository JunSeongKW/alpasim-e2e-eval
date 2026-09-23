#!/usr/bin/env bash
# One clip, one driver: does the corridor gate actually fire end to end?
#
# Every failure of this feature so far has been silent -- a whitelist that
# dropped a key, a filter applied after the candidates were already chosen --
# so the cheap check before committing another full arm is to run a single clip
# and look for the gate's own log line. No line, no gate.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_cache_filter"
AXE="$ROOT/e2e_challenge/axe_local_eval"
IMG=alpasim-e2e-drivesuprim-stage3:merged-route-ep30
RUN_DIR="$ROOT/runs/gateverify"
rm -rf "$RUN_DIR"
cd "$ROOT"

addrs="$(IMAGE="$IMG" CONTAINER_PREFIX=axe-gv BASE_PORT=6980 \
    GPU_INDICES_CSV=4 REPLICAS_PER_GPU=1 ROUTE_CACHE=1 \
    "$HERE/start_drivers_routecache.sh" 2>&1 | tail -1)"
[[ "$addrs" == \[* ]] || { echo "drivers failed: $addrs"; exit 1; }
echo "drivers: $addrs"

RUN_DIR="$RUN_DIR" PRESET=dev CONTESTANT_IMAGE="$IMG" DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$HERE/clips1.txt" N_ROLLOUTS=1 ROLLOUT_WORKERS=1 \
RENDER_GPUS_CSV=4,5 RENDERER_REPLICAS_PER_GPU=1 NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=false SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=false KEEP_ROLLOUTS=1 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" > "$RUN_DIR.wizard.log" 2>&1
echo "wizard exit=$?"

n="$(docker logs axe-gv-g4-0 2>&1 | grep -c ROUTEGATE)"
echo "=== ROUTEGATE lines: $n ==="
docker logs axe-gv-g4-0 2>&1 | grep ROUTEGATE | tail -5
docker logs axe-gv-g4-0 > "$RUN_DIR.driver.log" 2>&1 || true
docker rm -f axe-gv-g4-0 >/dev/null 2>&1 || true
docker ps -aq --filter 'name=gateverify' | xargs -r docker rm -f >/dev/null 2>&1 || true
