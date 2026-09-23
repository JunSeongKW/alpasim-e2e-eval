#!/usr/bin/env bash
# Wait for the running baseline evaluation to finish, then repeat it with the
# ego-footprint driver and aggregate both.
#
# Detached on purpose: the baseline has hours left, and the follow-up must not
# depend on a terminal staying open.
set -uo pipefail

ROOT="/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim"
BASE_RUN="$ROOT/runs/axe-ep24-curatedval-dev"
EGO_RUN="$ROOT/runs/axe-ep24-curatedval-egobox"
LOG_DIR="$ROOT/runs/_chain_logs"
IMAGE="alpasim-e2e-drivesuprim-stage3:ep24-egobox"
SHA="1d9fa8d7b8bb4f6d5f732cdf203a3a44c603780c1086726be841eb050d1212dc"
DRV_PREFIX="axe-egobox-drv"
BASE_PORT=6960

mkdir -p "$LOG_DIR"
say() { echo "[$(date '+%F %T')] $*" ; }

say "waiting for the baseline run to finish"
while pgrep -af "alpasim/.venv/bin/alpasim_wizard" | grep -q "axe-ep24-curatedval-dev"; do
    sleep 60
done
say "baseline wizard exited"

# A finished run leaves a summary; anything else means it died and the
# comparison would be against a partial baseline.
for _ in $(seq 1 20); do
    [[ -f "$BASE_RUN/aggregate/results-summary.json" ]] && break
    sleep 30
done
if [[ ! -f "$BASE_RUN/aggregate/results-summary.json" ]]; then
    say "ERROR: baseline produced no results-summary.json; stopping"
    exit 1
fi
say "baseline summary present"

say "tearing down baseline stack and drivers"
docker ps -a --format '{{.Names}}' | grep -E '^axe-ep24-curatedval-dev-' | xargs -r docker rm -f >/dev/null 2>&1
docker ps -a --format '{{.Names}}' | grep -E '^axe-curatedval-drv-' | xargs -r docker rm -f >/dev/null 2>&1
sleep 20

say "starting 16 ego-footprint drivers on GPUs 4-7"
DRIVERS=$(
    IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" \
    DRIVESUPRIM_EGO_FOOTPRINT_FROM_API=1 DRIVESUPRIM_EGO_CENTER_OFFSET=1 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 BASE_PORT="$BASE_PORT" \
    CONTAINER_PREFIX="$DRV_PREFIX" READY_TIMEOUT_SEC=2400 \
    "$ROOT/e2e_challenge/axe_local_eval/start_drivers.sh" 2>&1 | tee "$LOG_DIR/egobox_drivers.log" | tail -1
)
if [[ "$DRIVERS" != \[* ]]; then
    say "ERROR: drivers did not come up; see $LOG_DIR/egobox_drivers.log"
    exit 1
fi
say "drivers ready: $DRIVERS"

say "running curated_val with the ego-footprint driver"
cd "$ROOT" || exit 1
PRESET=dev DRIVER_ADDRESSES="$DRIVERS" RUN_NAME=axe-ep24-curatedval-egobox \
N_ROLLOUTS=3 ROLLOUT_WORKERS=16 RENDER_GPUS_CSV=4,5,6,7 \
RENDERER_REPLICAS_PER_GPU=4 NRE_CACHE_SIZE=1 \
    "$ROOT/e2e_challenge/axe_local_eval/run_curated_val.sh" > "$LOG_DIR/egobox_eval.log" 2>&1
say "ego-footprint run finished (exit $?)"

if [[ -f "$EGO_RUN/aggregate/results-summary.json" ]]; then
    say "summary written: $EGO_RUN/aggregate/results-summary.json"
else
    say "WARNING: no summary for the ego-footprint run"
fi

say "chain complete"
