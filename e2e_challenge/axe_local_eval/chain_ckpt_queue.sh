#!/usr/bin/env bash
# Evaluate a queue of checkpoints back to back, one 441-clip run each, under the
# axe-v9 contract. Builds each overlay image itself, so nothing has to be
# prepared in advance, and never leaves the cards idle between runs.
#
# Each entry is  TAG|CHECKPOINT_PATH|SHA256|BASE_PORT|WIZARD_BASEPORT
# and every run goes through run_ckpt_441.sh: axe-v9 base with only the weights
# replaced, OFFICIAL_ENV_ONLY drivers (no reranker files mounted), dev preset,
# 441 curated_val, one rollout, 16 drivers / 16 renderers / 16 workers, gains
# lat 1.0 / lon 0.25 / idx 3, no video.
#
# It waits for whatever run_ckpt_441.sh run is already going before starting the
# first of its own -- all four cards are needed, and the launcher refuses to
# start unless they are nearly empty. The wait is on the log line the launcher
# writes when it finishes, not on a process, so this survives the session that
# started it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AXE="$ROOT/e2e_challenge/axe_local_eval"
cd "$ROOT"
LOG="$AXE/ckpt_queue.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# The run already in flight, if any: wait for its OK/FAILED line.
WAIT_FOR="${WAIT_FOR:-}"
if [[ -n "$WAIT_FOR" && -f "$AXE/$WAIT_FOR.log" ]]; then
    log "waiting for $WAIT_FOR to finish"
    until grep -qE '^\[.*\] (OK ->|FAILED)' "$AXE/$WAIT_FOR.log" 2>/dev/null; do sleep 300; done
    log "$WAIT_FOR: $(grep -E '^\[.*\] (OK ->|FAILED)' "$AXE/$WAIT_FOR.log" | tail -1 | cut -c1-110)"
fi

QUEUE=(
  "disjoint-ep04|$ROOT/../models/stage3_disjoint_ep04.ckpt|3ce4eb8815ccaef163194a244a4afec8881ef56ea85e69aa66c12edc4c1f8ee4|7800|20300"
  "disjoint-ep29|$ROOT/../models/stage3_disjoint_ep29.ckpt|cb6205d5209abeed181278490189fea87b03bb9e2c4e94939ccc65508eb1442c|7900|20500"
)

for entry in "${QUEUE[@]}"; do
    IFS='|' read -r tag ckpt sha port wiz <<< "$entry"
    img="alpasim-e2e-axe-v9:${tag}"
    log "=== $tag: build overlay image ==="
    ctx="$(mktemp -d)"
    trap 'rm -rf "$ctx"' EXIT
    cp "$ckpt" "$ctx/checkpoint.ckpt" || { log "$tag: checkpoint missing at $ckpt"; continue; }
    cp "$AXE/Dockerfile.checkpoint_overlay" "$ctx/Dockerfile"
    if ! docker build -t "$img" \
            --build-arg BASE_IMAGE=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9 \
            --build-arg CHECKPOINT_SHA256="$sha" \
            --build-arg CHECKPOINT_SOURCE="$ckpt" "$ctx" >>"$LOG" 2>&1; then
        log "$tag: build FAILED, skipping"; rm -rf "$ctx"; continue
    fi
    rm -rf "$ctx"
    label="$(docker image inspect "$img" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null)"
    [[ "$label" == "$sha" ]] || { log "$tag: image label $label != $sha, skipping"; continue; }
    log "$tag: image ok"

    # One driver first: a checkpoint whose tensors do not match this architecture
    # would otherwise be found six hours later, and the load line is the only
    # place that shows it.
    log "=== $tag: smoke test one driver ==="
    if IMAGE="$img" EXPECTED_CHECKPOINT_SHA256="$sha" OFFICIAL_ENV_ONLY=1 \
            CONTAINER_PREFIX="smoke-$tag" BASE_PORT=$((port + 90)) \
            GPU_INDICES_CSV=0 REPLICAS_PER_GPU=1 READY_TIMEOUT_SEC=900 \
            "$AXE/start_drivers.sh" >>"$LOG" 2>&1; then
        docker logs "smoke-$tag-g0-0" 2>&1 | grep -E 'checkpoint load|policy load complete' | tee -a "$LOG"
    else
        log "$tag: smoke FAILED, skipping"
    fi
    docker rm -f "smoke-$tag-g0-0" >/dev/null 2>&1 || true

    # The cards have to come back before the launcher will start.
    for _ in $(seq 1 20); do
        busy=0
        for mib in $(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits); do
            (( mib < 4000 )) || busy=1
        done
        (( busy == 0 )) && break
        sleep 30
    done

    log "=== $tag: 441 clips ==="
    : > "$AXE/$tag.log"
    start=$(date +%s)
    TAG="$tag" IMG="$img" SHA="$sha" CKPT="$ckpt" \
    BASE_PORT="$port" WIZARD_BASEPORT="$wiz" GPUS=0,1,2,3 \
        "$AXE/run_ckpt_441.sh" > "$AXE/$tag.nohup" 2>&1
    log "$tag: launcher exit=$? after $((($(date +%s) - start) / 60)) min"
    nohup "$AXE/progress_ckpt_441.sh" "$tag" >/dev/null 2>&1 &

    d="$ROOT/runs/leaderboard-$tag/aggregate/results-summary.json"
    if [[ -f "$d" ]]; then
        log "$tag: summary ok"
        "$ROOT/.venv/bin/python" "$AXE/compare_on_clips.py" \
            "$tag=$d" \
            "axe-v9=$ROOT/runs/leaderboard-merged-route-ep30/aggregate/results-summary.json" \
            2>&1 | tee -a "$LOG"
    else
        log "$tag: NO summary"
    fi
    for c in $(docker ps -a --format '{{.Names}}' | grep -E "^(axe-lb-$tag|leaderboard-$tag)" | sort); do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
done
log "=== queue done ==="
