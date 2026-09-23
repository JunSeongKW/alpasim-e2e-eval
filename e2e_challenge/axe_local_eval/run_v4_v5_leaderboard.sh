#!/usr/bin/env bash
# Run the axe-v4 and axe-v5 submission images on curated_val, then fit the local
# Drive-IRT leaderboard over them together with axe-v8 and the organizer bundle.
#
# Preset is dev, not ec2, and that is forced by the fit rather than chosen:
# ZOIB needs S + 5N observations for S subjects and N scenes, which on the
# 441-scene split means six subjects minimum. Three of our own models fall back
# to an arithmetic average with no PCS and no rank intervals, so the eight
# published reference subjects have to join the fit -- and those were produced
# under dev. The existing axe-v8 run is dev as well.
#
# One rollout per scene instead of the references' three: the per-scene estimate
# is noisier, but the scene set is what has to match, and this turns an 8-hour
# pair of runs into something that finishes overnight.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
ECR=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe
LOG="$AXE/v4v5.log"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

run_subject() {
    local tag="$1" port="$2"
    local run_dir="$ROOT/runs/leaderboard-${tag}"

    if [[ -f "$run_dir/aggregate/results-summary.json" ]]; then
        log "$tag already has a summary; skipping"
        return 0
    fi

    log "=== $tag: starting 16 drivers ==="
    # v4/v5 predate the checkpoint label, so the sha guard is opted out of.
    local addrs
    # v4/v5 load the policy lazily: PolicyHandle has no eager start, and
    # "policy load complete" is only logged once the first session arrives.
    # Waiting for it before any session exists deadlocks, so readiness here is
    # the driver accepting gRPC; start_session then waits for the load, which is
    # how these images ran on the official leaderboard.
    addrs="$(IMAGE="$ECR:$tag" \
        EXPECTED_CHECKPOINT_SHA256= \
        OFFICIAL_ENV_ONLY=1 \
        READY_MARKER="driver listening" \
        CONTAINER_PREFIX="axe-lb-${tag}" \
        BASE_PORT="$port" \
        GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
        "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
    if [[ "$addrs" != \[* ]]; then
        log "$tag: drivers failed to start"
        return 1
    fi

    log "=== $tag: evaluating 441 scenes x 1 rollout (dev) ==="
    RUN_DIR="$run_dir" \
    PRESET=dev \
    CONTESTANT_IMAGE="$ECR:$tag" \
    DRIVER_ADDRESSES="$addrs" \
    N_ROLLOUTS=1 \
    ROLLOUT_WORKERS=16 \
    RENDER_GPUS_CSV=4,5,6,7 \
    RENDERER_REPLICAS_PER_GPU=4 \
    NRE_CACHE_SIZE=1 \
    ENABLE_AUTORESUME=true \
        "$AXE/run_curated_val.sh" >> "$run_dir.wizard.log" 2>&1
    log "$tag: wizard exit=$?"

    docker rm -f $(docker ps -aq --filter "name=axe-lb-${tag}") >/dev/null 2>&1 || true

    if [[ -f "$run_dir/aggregate/results-summary.json" ]]; then
        log "$tag: OK -> $run_dir/aggregate/results-summary.json"
    else
        log "$tag: FAILED, no summary produced"
        return 1
    fi
}

run_subject axe-v4 6820 || exit 1
run_subject axe-v5 6840 || exit 1

log "=== fitting the local leaderboard ==="
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
uv run --extra local-evaluation python "$ROOT/e2e_challenge/local_evaluation/evaluate.py" \
    --track pai \
    --run axe-v4="$ROOT/runs/leaderboard-axe-v4" \
    --run axe-v5="$ROOT/runs/leaderboard-axe-v5" \
    --run axe-v8="$ROOT/runs/axe-ep24-curatedval-egobox" \
    --output-dir "$ROOT/runs/leaderboard-axe-v4v5v8" \
    --device cpu >> "$LOG" 2>&1

log "=== done ==="
column -s, -t "$ROOT/runs/leaderboard-axe-v4v5v8/capability_ranking.csv" 2>&1 | tee -a "$LOG"
