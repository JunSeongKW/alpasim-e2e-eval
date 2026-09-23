#!/usr/bin/env bash
# Verification run: the experiment-1 + experiment-2 winners on the full 10-clip
# set, so the tuning can be checked against clips it was NOT selected on.
#
#   MPC     lat_first : long 0.5, lat 6.0, idx_start 3
#   ranking dac3      : drivable_area_compliance exponent 3.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-flatvits-nodistill:ep19}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-9f43a00619edfec15e5f44b5aa8942c312e382e5cd0fce8f756b270e44a22ef2}"
# Must match the checkpoint; the driver refuses a mismatch.
export DRIVESUPRIM_BACKBONE_TYPE="${DRIVESUPRIM_BACKBONE_TYPE:-vits_flat}"
export DRIVESUPRIM_CONFIG_OVERRIDE="${DRIVESUPRIM_CONFIG_OVERRIDE:-$SCRIPT_DIR/assets/drivesuprim/flatvits_nodistill_ep19_eval/rank_variants/dac3.json}"

MPC_OVERRIDES="${MPC_OVERRIDES:-controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3}"
CLIPS_FILE="${CLIPS_FILE:-$SCRIPT_DIR/flatvits_10clips_seed20260903.txt}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/e2e_challenge_drivesuprim_debug/bestcombo_10clip_$(date +%H%M%S)}"

# Ports, container prefix and worker count are parameterised so two configs can
# be verified side by side on the same box.
PORTS_A="${PORTS_A:-6940,6941,6942,6943}"
PORTS_B="${PORTS_B:-6950,6951,6952,6953}"
# One GPU per port within a group; the two lists must be the same length.
GPUS_A="${GPUS_A:-0,1,2,3}"
GPUS_B="${GPUS_B:-0,1,2,3}"
TAG="${TAG:-bc}"
WORKERS="${WORKERS:-8}"
BASEPORT="${BASEPORT:-6400}"
RENDER_GPUS="${RENDER_GPUS:-0,1,2,3,0,1,2,3}"

mkdir -p "$RUN_DIR/_logs"
pids=()
# Container names are <prefix>-gpu<N>, so two prefixes give eight replicas
# across four GPUs.
for grp in "a:$PORTS_A:$GPUS_A" "b:$PORTS_B:$GPUS_B"; do
    prefix="${TAG}-${grp%%:*}"; rest="${grp#*:}"; ports="${rest%%:*}"; gpus="${rest#*:}"
    GPU_INDICES_CSV="$gpus" DRIVER_PORTS_CSV="$ports" CONTAINER_PREFIX="$prefix" \
    DRIVER_LOG_FILE="$RUN_DIR/_logs/driver_${prefix}.log" \
        "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh" \
        > "$RUN_DIR/_logs/stack_${prefix}.log" 2>&1 &
    pids+=($!)
done

for _ in $(seq 1 180); do
    ready=$(cat "$RUN_DIR"/_logs/driver_*.log 2>/dev/null | grep -c 'policy load complete') || ready=0
    [[ "$ready" -ge "$WORKERS" ]] && break
    sleep 5
done
echo "drivers ready: $ready/$WORKERS"

RUN_DIR="$RUN_DIR" SCENE_IDS_CSV="$(paste -sd, "$CLIPS_FILE")" \
DRIVER_PORTS_CSV="${PORTS_A},${PORTS_B}" \
RENDER_GPUS_CSV="$RENDER_GPUS" ROLLOUT_WORKERS="$WORKERS" NRE_CACHE_SIZE=1 \
RENDER_VIDEO=true WIZARD_BASEPORT="$BASEPORT" SERVICE_STARTUP_TIMEOUT_SEC=2400 \
EXTRA_OVERRIDES="$MPC_OVERRIDES" \
    "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_debug_video_eval.sh" \
    > "$RUN_DIR/_logs/eval.log" 2>&1 || echo "eval exited $?"

for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
docker rm -f $(docker ps -aq --filter "name=${TAG}-") 2>/dev/null || true
echo "best-combo verification done: $RUN_DIR"
