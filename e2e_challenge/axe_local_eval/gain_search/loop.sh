#!/usr/bin/env bash
# Run the controller-gain search until someone stops it.
#
# One iteration: ask search.py for the next gain set, evaluate it on the
# stratified SEARCH subset, score it with the weighted estimator, append the
# record. Nothing is kept in memory between iterations, so killing this script
# at any point loses at most the run in flight -- the next start re-proposes it.
#
# The 16 driver replicas are external and stay up across iterations; only the
# renderer/physics/controller stack is rebuilt per run, which is roughly eight
# minutes of the hour each iteration costs.
#
# Stop with: touch <this dir>/STOP     (finishes the run in flight first)
#        or: pkill -f gain_search/loop.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

# search.py reads the same file to pick the incumbent, so both must agree; the
# env var keeps one model's history out of another's search.
RESULTS="${GAIN_RESULTS:-$HERE/results.jsonl}"
LOG="${GAIN_LOG:-$HERE/loop.log}"
mkdir -p "$(dirname "$RESULTS")" "$(dirname "$LOG")"
DRIVER_ADDRESSES="${DRIVER_ADDRESSES:?set DRIVER_ADDRESSES to the hydra list of driver endpoints}"
SUBSET="${SUBSET:-search}"
# Gains are evaluated by the organizers under the ec2 preset, so tune under it:
# a gain set is only worth what it scores in the environment that will score it.
# It is also about 2.8x faster than dev here (5.4 vs 1.9 rollout/min), because
# the driver is not called during force-GT and RGB renders are batched.
PRESET="${PRESET:-ec2}"
CONTESTANT_IMAGE="${CONTESTANT_IMAGE:-alpasim-e2e-drivesuprim-stage3:submit-v1}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
# Which GPUs the renderer/physics stack lands on. The drivers are external and
# pinned separately by start_drivers.sh, so both have to agree.
RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-4,5,6,7}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs/gain-search}"
mkdir -p "$RUNS_ROOT"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

MIN_FREE_GB="${MIN_FREE_GB:-200}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"
consecutive_failures=0

while true; do
    if [[ -f "$HERE/STOP" ]]; then
        log "STOP file present; exiting"
        exit 0
    fi

    # A full disk kills the runtime mid-rollout and every later config fails the
    # same way, so stop before that rather than burning hours on it.
    free_gb=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9')
    if (( free_gb < MIN_FREE_GB )); then
        log "only ${free_gb} GB free (need ${MIN_FREE_GB}); exiting"
        exit 1
    fi

    if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
        log "${consecutive_failures} consecutive failures; exiting"
        exit 1
    fi

    next="$(uv run python "$HERE/search.py" next)" || {
        log "search exhausted: $next"
        exit 0
    }
    tag="$(jq -r .tag <<<"$next")"
    why="$(jq -r .why <<<"$next")"
    gains="$(jq -c .gains <<<"$next")"

    overrides=""
    for k in $(jq -r 'keys[]' <<<"$gains"); do
        v="$(jq -r --arg k "$k" '.[$k]' <<<"$gains")"
        overrides+="controller.gains.${k}=${v} "
    done

    run_dir="$RUNS_ROOT/$tag"
    log "=== $tag ==="
    log "    $why"
    log "    gains: $gains"
    log "    preset=$PRESET image=$CONTESTANT_IMAGE n_rollouts=$N_ROLLOUTS subset=$SUBSET"

    if [[ -d "$run_dir" ]]; then
        # A half-finished directory from a killed iteration would be resumed
        # with the previous gains baked into its generated config.
        log "    removing stale run dir"
        rm -rf "$run_dir"
    fi

    RUN_DIR="$run_dir" \
    PRESET="$PRESET" \
    CONTESTANT_IMAGE="$CONTESTANT_IMAGE" \
    DRIVER_ADDRESSES="$DRIVER_ADDRESSES" \
    SCENE_IDS_FILE="$HERE/${SUBSET}_clips.txt" \
    N_ROLLOUTS="$N_ROLLOUTS" \
    ROLLOUT_WORKERS=16 \
    RENDER_GPUS_CSV="$RENDER_GPUS_CSV" \
    RENDERER_REPLICAS_PER_GPU=4 \
    NRE_CACHE_SIZE=1 \
    ENABLE_AUTORESUME=false \
    MPC_OVERRIDES="$overrides" \
        "$ROOT/e2e_challenge/axe_local_eval/run_curated_val.sh" \
        >> "$run_dir.wizard.log" 2>&1
    rc=$?
    log "    wizard exit=$rc"

    # Defensive: the wizard normally tears its own stack down, but a crash can
    # leave containers holding the GPUs and the next iteration would then fail
    # to start renderers.
    if [[ -f "$run_dir/docker-compose.yaml" ]]; then
        docker compose -f "$run_dir/docker-compose.yaml" down --remove-orphans \
            >/dev/null 2>&1 || true
    fi

    scored="$(uv run --package alpasim-eval python "$HERE/score_run.py" \
        --run "$run_dir" --subset "$SUBSET" --tmp "$run_dir/.score" 2>>"$LOG")"
    if [[ -z "$scored" ]] || ! jq -e .score <<<"$scored" >/dev/null 2>&1; then
        log "    SCORING FAILED, recording as failed"
        jq -c -n --arg tag "$tag" --argjson gains "$gains" --arg why "$why" \
            '{tag:$tag, gains:$gains, why:$why, score:null, error:"scoring failed"}' \
            >> "$RESULTS"
        rm -rf "$run_dir/rollouts" "$run_dir/.score"
        consecutive_failures=$((consecutive_failures + 1))
        continue
    fi
    consecutive_failures=0

    jq -c -n --arg tag "$tag" --argjson gains "$gains" --arg why "$why" \
        --argjson s "$scored" --arg at "$(date -Is)" \
        '{tag:$tag, gains:$gains, why:$why, at:$at} + $s' >> "$RESULTS"

    # Each run writes ~49 GB of rollout trajectories and only the score is ever
    # read back. Keeping them filled a 9.7 TB disk after 63 runs, which killed
    # the runtime mid-run and cascaded into 52 failed configs.
    rm -rf "$run_dir/rollouts" "$run_dir/.score"

    log "    score=$(jq -r .score <<<"$scored") se=$(jq -r .se <<<"$scored") \
clips=$(jq -r .clips <<<"$scored") zero=$(jq -r .zero_rate <<<"$scored")"
done
