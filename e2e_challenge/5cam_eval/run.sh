#!/usr/bin/env bash
# Fixed, comparable 441 evaluation. --check is read-only; --wait waits for 4-7.
# Run from any working directory. stdout belongs in runs/<name>.progress.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
MODE="${1:-run}"
[[ "$MODE" == run || "$MODE" == --check || "$MODE" == --wait || "$MODE" == --resume ]] || {
    echo 'Usage: run.sh [--check | --wait | --resume]' >&2; exit 2;
}
IMAGE='alpasim-e2e-drivesuprim-stage3:5cam-ep05-20261007'
SHA='79c625f3a29c49da6b9605b1062de5e8e6ec5da8391362b25507e35725bb8b8e'
RUN_NAME='leaderboard-stage3-5cam-ep05-20261007'
RUN_DIR="$ROOT/runs/$RUN_NAME"
PREP="$ROOT/runs/prepare-stage3-5cam-ep05"
PREFIX='axe-5cam-ep05-drv'
GPUS='4,5,6,7'
AXE="$ROOT/e2e_challenge/axe_local_eval"
export UV_OFFLINE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
[[ "$image_id" == "$(cat "$PREP/image-id.txt")" ]] || { log 'Image changed since CPU validation'; exit 2; }
[[ "$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}')" == "$SHA" ]]
"$ROOT/.venv/bin/python" - "$PREP" "$SHA" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); report=json.loads((p/'cpu_validation.json').read_text())
assert report['passed'] and report['checkpoint_sha256']==sys.argv[2]
assert json.loads((p/'scene_audit.json').read_text())['scene_count']==441
PY
"$ROOT/.venv/bin/python" "$HERE/leaderboard.py" --check-existing
log 'Prepared: 441 scenes; dev; 16 drivers/workers/renderers; lat/lon/idx=1/0.25/3'
if [[ "$MODE" == --check ]]; then
    nvidia-smi -i "$GPUS" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    exit 0
fi
mkdir -p "$ROOT/.cache"
exec 9>"$ROOT/.cache/stage3-5cam-441.lock"
flock -n 9 || { log 'Another 5cam launcher is already active'; exit 2; }
if [[ "$MODE" != --resume && -e "$RUN_DIR" ]]; then
    log "Refusing to overwrite $RUN_DIR; use --resume for an interrupted run"; exit 2
fi
gpu_free() {
    local result
    result="$(nvidia-smi -i "$GPUS" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" || return 1
    [[ "$(wc -l <<< "$result")" == 4 ]] || return 1
    while IFS=, read -r used utilization; do
        (( used < 4000 && utilization <= 5 )) || return 1
    done <<< "$result"
}
if [[ "$MODE" == --wait ]]; then
    log "Waiting for all GPUs $GPUS: used VRAM <4000 MiB, utilization <=5%; no jobs will be stopped"
    stable=0
    while (( stable < 2 )); do
        if gpu_free; then stable=$((stable + 1)); else stable=0; fi
        (( stable == 2 )) || sleep 30
    done
fi
gpu_free || { log "GPUs $GPUS are occupied; retry later or use --wait"; exit 75; }
available_kib="$(df -Pk "$ROOT/runs" | awk 'NR==2 {print $4}')"
(( available_kib >= 262144000 )) || { log 'Need at least 250 GiB free to retain the rollouts'; exit 2; }
# Refuse existing containers, even stopped ones, before start_drivers can remove them.
existing="$(docker ps -a --filter "name=^${PREFIX}-g[4567]-[0-3]$" --format '{{.Names}}')"
[[ -z "$existing" ]] || { log "Driver names already exist: $existing"; exit 2; }
"$ROOT/.venv/bin/python" - <<'PY'
import socket
for port in range(7160,7176):
    with socket.socket() as s:
        s.bind(('127.0.0.1',port))
print('PASS: all 16 driver ports are available')
PY
mkdir -p "$RUN_DIR"
if [[ "$MODE" == --resume ]]; then
    "$ROOT/.venv/bin/python" - "$RUN_DIR" "$image_id" "$SHA" <<'PY'
import json,sys
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'evaluation_provenance.json').read_text())
assert p['image_id']==sys.argv[2] and p['checkpoint_sha256']==sys.argv[3]
assert p['preset']=='dev' and p['scenes']==441 and p['rollouts_per_scene']==1
PY
else
    "$ROOT/.venv/bin/python" - "$RUN_DIR" "$image_id" "$SHA" <<'PY'
import json,subprocess,sys
from pathlib import Path
p={'subject':'stage3-5cam-ep05','checkpoint':'stage3_5cam_ep05.ckpt',
   'checkpoint_sha256':sys.argv[3],'image_id':sys.argv[2],
   'camera_order':['CAM_L1','CAM_L0','CAM_F0','CAM_R0','CAM_R1'],
   'point_cloud_range':[-56,-28,-3,56,28,5],'bev_grid_hw':[56,112],
   'preset':'dev','scenes':441,'rollouts_per_scene':1,'gpus':[4,5,6,7],
   'drivers':16,'workers':16,'renderers':16,'route_reranker':False,
   'official_env_only':True,'ego_footprint_from_api':True,'ego_center_offset':True,
   'mpc':{'lat_position_weight':1.0,'long_position_weight':0.25,'idx_start_penalty':3},
   'keep_rollouts':True,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
(Path(sys.argv[1])/'evaluation_provenance.json').write_text(json.dumps(p,indent=2)+'\n')
PY
fi
stop_our_drivers() {
    mkdir -p "$RUN_DIR/driver-logs"
    for gpu in 4 5 6 7; do
        for replica in 0 1 2 3; do
            name="${PREFIX}-g${gpu}-${replica}"
            docker logs "$name" > "$RUN_DIR/driver-logs/$name.log" 2>&1 || true
        done
    done
    CONTAINER_PREFIX="$PREFIX" "$AXE/stop_drivers.sh" || true
}
cleanup() {
    local status=$?
    trap - EXIT
    if [[ "${drivers_stopped:-0}" != 1 ]]; then stop_our_drivers; fi
    log "Launcher exit=$status; raw rollouts retained at $RUN_DIR"
    exit "$status"
}
trap cleanup EXIT
if ! git rev-parse "refs/tags/run/$RUN_NAME" >/dev/null 2>&1; then
    git tag "run/$RUN_NAME"
fi
log "Starting 16 five-camera drivers on GPUs $GPUS"
IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" OFFICIAL_ENV_ONLY=1 \
    GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU=4 BASE_PORT=7160 \
    CONTAINER_PREFIX="$PREFIX" READY_TIMEOUT_SEC=1800 \
    "$AXE/start_drivers.sh" > "$RUN_DIR/driver-start.log" 2>&1
addrs="$(tail -1 "$RUN_DIR/driver-start.log")"
[[ "$addrs" == \[* ]] || { log 'Drivers did not return their addresses'; exit 1; }
log "Starting $RUN_NAME; simulator log: $RUN_DIR.wizard.log"
# Clear subset/video overrides so inherited shell settings cannot change the contract.
RUN_NAME="$RUN_NAME" RUN_DIR="$RUN_DIR" PRESET=dev CONTESTANT_IMAGE="$IMAGE" \
    DRIVER_ADDRESSES="$addrs" N_ROLLOUTS=1 ROLLOUT_WORKERS=16 \
    RENDER_GPUS_CSV="$GPUS" RENDERER_REPLICAS_PER_GPU=4 NRE_CACHE_SIZE=1 \
    SCENE_LIMIT=0 SCENE_IDS_FILE='' RENDER_VIDEO=false KEEP_ROLLOUTS=1 \
    ENABLE_AUTORESUME=true SERVICE_STARTUP_TIMEOUT_SEC=1800 FAST_STARTUP=1 \
    ALPASIM_IMAGE=nvcr.io/nvidia/nre/nre-ga:26.04 NRE_IMAGE=nvcr.io/nvidia/nre/nre-ga:26.04 \
    DRIVER_CONCURRENT_ROLLOUTS=1 \
    MPC_OVERRIDES='controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3' \
    EXTRA_OVERRIDES='wizard.baseport=19900' \
    "$AXE/run_curated_val.sh" > "$RUN_DIR.wizard.log" 2>&1
log 'Simulation finished; validating all 441 results and fitting all 35 subjects on CPU'
stop_our_drivers
drivers_stopped=1
"$ROOT/.venv/bin/python" "$HERE/leaderboard.py" --run-dir "$RUN_DIR"
log 'Complete: runs/leaderboard-35-with-stage3-5cam-ep05/local_leaderboard.csv'
