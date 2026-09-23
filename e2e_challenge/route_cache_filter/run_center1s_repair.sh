#!/usr/bin/env bash
# Re-run only the full-run scenes that hit the RouteCache singleton-update bug.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_cache_filter"
AXE="$ROOT/e2e_challenge/axe_local_eval"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/runs/routecache-val441-center1s}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/routecache-val441-center1s-repair}"
CLIPS="${CLIPS:-$HERE/center1s_repair_clips.txt}"
LOG="${LOG:-$HERE/center1s_repair.log}"
IMG="${IMG:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"

cd "$ROOT"
python3 - "$SOURCE_RUN/aggregate/results-summary.json" "$CLIPS" <<'PY'
import json
import sys

source, output = sys.argv[1:]
rows = json.load(open(source))["rollouts"]
scene_ids = [
    row["clipgt_id"] for row in rows
    if "index 1 is out of bounds for axis 0 with size 1"
    in str(row.get("failure_reason", ""))
]
with open(output, "w") as f:
    f.write("\n".join(scene_ids) + ("\n" if scene_ids else ""))
print(f"repair scenes: {len(scene_ids)}")
PY

count="$(wc -l < "$CLIPS")"
[[ "$count" -gt 0 ]] || { echo "No repair scenes"; exit 0; }
rm -rf "$RUN_DIR"
echo "[$(date '+%F %T')] starting $count repair scenes" | tee "$LOG"

addrs="$(IMAGE="$IMG" \
    CONTAINER_PREFIX=axe-rc441-center1s-repair \
    BASE_PORT=6910 \
    GPU_INDICES_CSV=4,5,6 REPLICAS_PER_GPU=2 \
    DRIVER_PKG="$HERE/drivesuprim_challenge" \
    MODEL_PY="$HERE/drivesuprim_model.py" \
    AGENT_PY="$HERE/drivesuprim_agent.py" \
    ROUTE_CACHE=1 ROUTE_CORRIDOR_M=4.0 ROUTE_GATE_HORIZON_S=1 \
    "$HERE/start_drivers_routecache.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { echo "drivers failed" | tee -a "$LOG"; exit 1; }
echo "[$(date '+%F %T')] drivers: $addrs" | tee -a "$LOG"

set +e
RUN_DIR="$RUN_DIR" \
UV_OFFLINE=1 \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$CLIPS" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=5 \
RENDER_GPUS_CSV=0,1,2,3,7 \
RENDERER_REPLICAS_PER_GPU=1 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=false \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=false \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
status=$?
set -e
echo "[$(date '+%F %T')] wizard exit=$status" | tee -a "$LOG"

for c in $(docker ps -aq --filter name=axe-rc441-center1s-repair); do
    docker logs "$c" > "$RUN_DIR.driver-$c.log" 2>&1 || true
done
docker ps -aq --filter name=axe-rc441-center1s-repair | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
    echo "[$(date '+%F %T')] OK: $RUN_DIR/aggregate/results-summary.json" | tee -a "$LOG"
else
    echo "[$(date '+%F %T')] FAILED: no summary" | tee -a "$LOG"
    exit "${status:-1}"
fi
