#!/usr/bin/env bash
# Re-run the submitted axe-v9 image on the 441 curated_val clips, keeping the
# per-step metrics, and compare its speed at curves with the human log.
#
#   nohup e2e_challenge/axe_local_eval/run_curve_axe_v9.sh 20261010 \
#       > runs/curve-axe-v9-441-20261010.progress.log 2>&1 &
#
# The scored axe-v9 run (runs/leaderboard-merged-route-ep30) kept only
# results-summary.json, which has one row per rollout and no speed trace. This is
# the same run again -- same image, same contract as run_ckpt_441.sh: dev preset,
# 441 scenes x 1, 16 drivers / 16 workers / 16 renderers on GPUs 4-7, gains
# lat 1.0 / lon 0.25 / idx 3, OFFICIAL_ENV_ONLY -- so the per-clip scores can be
# checked against the scored run before the speed traces are trusted.
#
# Two differences from run_ckpt_441.sh, neither touching what is simulated:
#   * it waits for GPUs 4-7 to be empty instead of refusing, because another
#     researcher's job holds them now and a 16-renderer stack beside it would
#     push that job out of memory;
#   * rollout.asl (~400 MB each) is removed as soon as the rollout's own
#     metrics.parquet is written, instead of ~175 GB at the end -- a bulk delete
#     on this volume stalls the journal for everyone else's jobs.
set -uo pipefail
DATE="${1:?date tag, e.g. 20261010}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
RUN_DIR="$ROOT/runs/curve-axe-v9-441-$DATE"
OUT="$ROOT/runs/curve-stop-vs-human-$DATE"
IMG=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9
SHA=364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d
GPUS=4,5,6,7
PREFIX="axe-curve-v9"
FREE_MIB="${FREE_MIB:-10000}"
log() { echo "[$(date '+%F %T')] $*"; }

[[ -f "$RUN_DIR/aggregate/metrics_unprocessed.parquet" ]] && { log "already done: $RUN_DIR"; SKIP_RUN=1; }

if [[ -z "${SKIP_RUN:-}" ]]; then
    # Empty means every card below FREE_MIB on two looks five minutes apart:
    # a job between phases can drop its memory for a moment.
    quiet=0
    while (( quiet < 2 )); do
        mapfile -t used < <(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits)
        busy=0; for mib in "${used[@]}"; do (( mib < FREE_MIB )) || busy=1; done
        if (( busy )); then
            (( quiet > 0 || ${waited:-0} % 6 == 0 )) && log "waiting: GPUs $GPUS used MiB ${used[*]}"
            quiet=0; waited=$(( ${waited:-0} + 1 ))
        else
            quiet=$((quiet + 1))
        fi
        (( quiet < 2 )) && sleep 300
    done
    log "GPUs $GPUS free (used MiB ${used[*]})"

    actual="$(docker image inspect "$IMG" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}')"
    [[ "$actual" == "$SHA" ]] || { log "image checkpoint mismatch: $actual"; exit 2; }
    mkdir -p "$RUN_DIR"
    cat > "$RUN_DIR/evaluation_provenance.json" <<EOF
{
  "purpose": "per-step speed traces of the submitted axe-v9 policy (curve near-stop vs human)",
  "image": "$IMG",
  "image_id": "$(docker image inspect "$IMG" --format '{{.Id}}')",
  "checkpoint_sha256": "$SHA",
  "scored_run": "runs/leaderboard-merged-route-ep30",
  "preset": "dev", "scenes": 441, "rollouts_per_scene": 1,
  "gpus": "$GPUS", "drivers": 16, "workers": 16, "renderer_replicas_per_gpu": 4,
  "official_env_only": true,
  "mpc": {"long_position_weight": 0.25, "lat_position_weight": 1.0, "idx_start_penalty": 3},
  "rollout_asl": "removed per rollout once metrics.parquet exists",
  "git_commit": "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

    log "starting 16 drivers on GPUs $GPUS"
    addrs="$(IMAGE="$IMG" EXPECTED_CHECKPOINT_SHA256="$SHA" OFFICIAL_ENV_ONLY=1 \
        CONTAINER_PREFIX="$PREFIX" BASE_PORT=7700 \
        GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU=4 \
        "$AXE/start_drivers.sh" 2>>"$RUN_DIR.drivers.log" | tail -1)"
    [[ "$addrs" == \[* ]] || { log "drivers failed (see $RUN_DIR.drivers.log)"; exit 1; }

    (   # prune each finished rollout's .asl; the per-step metrics stay
        while :; do
            find "$RUN_DIR/rollouts" -name _complete -mmin +1 2>/dev/null | while read -r done; do
                d="$(dirname "$done")"
                [[ -f "$d/rollout.asl" && -f "$d/metrics.parquet" ]] && rm -f "$d/rollout.asl"
            done
            sleep 60
        done
    ) &
    pruner=$!

    log "441 scenes x 1 rollout (dev), gains lat=1.0 lon=0.25 idx=3"
    start=$(date +%s)
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
    KEEP_ROLLOUTS=1 \
    MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    EXTRA_OVERRIDES="wizard.baseport=19900" \
        "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
    log "wizard exit=$? after $((($(date +%s) - start) / 60)) min"
    kill "$pruner" 2>/dev/null
    mapfile -t mine < <(docker ps -a --format '{{.Names}}' | grep -E "^${PREFIX}-")
    log "removing ${#mine[@]} driver containers (${PREFIX}-*)"
    (( ${#mine[@]} )) && docker rm -f "${mine[@]}" >/dev/null
    find "$RUN_DIR/rollouts" -name rollout.asl -delete 2>/dev/null
fi

[[ -f "$RUN_DIR/aggregate/metrics_unprocessed.parquet" ]] || { log "FAILED: no per-step metrics"; exit 1; }
log "reproduction check against the scored axe-v9 run"
"$ROOT/.venv/bin/python" - "$RUN_DIR" <<'EOF'
import json, sys
new = json.load(open(f"{sys.argv[1]}/aggregate/results-summary.json"))["rollouts"]
old = json.load(open("runs/leaderboard-merged-route-ep30/aggregate/results-summary.json"))["rollouts"]
old = {r["clipgt_id"]: r for r in old}
# The scored run has one clip whose metrics lack dist_traveled_m (an RPC failure).
pairs = [(r, old[r["clipgt_id"]]) for r in new if r["clipgt_id"] in old
         and "dist_traveled_m" in (r["metrics"] or {}) and "dist_traveled_m" in (old[r["clipgt_id"]]["metrics"] or {})]
same_score = sum(abs(a["score"] - b["score"]) < 1e-6 for a, b in pairs)
close_dist = sum(abs(a["metrics"]["dist_traveled_m"] - b["metrics"]["dist_traveled_m"]) < 1.0 for a, b in pairs)
mean = lambda rows: sum(r["score"] for r in rows) / len(rows)
print(f"clips {len(pairs)}: identical score {same_score}, dist within 1 m {close_dist}; "
      f"mean score rerun {mean([a for a, _ in pairs]):.4f} vs scored {mean([b for _, b in pairs]):.4f}")
EOF
R="--run axe-v9=$RUN_DIR"
for r in stage3-vits512-6layer-20261008 stage3-vitl512-3layer-lr1-20261008 stage3-vits768-3layer-20261008 \
         vits512-disjoint-baseline-ep04-20261007; do
    R="$R --run ${r%-2026*}=runs/leaderboard-$r"
done
"$ROOT/.venv/bin/python" "$AXE/curve_stop_vs_human.py" $R --out "$OUT"
log "done -> $OUT/summary.md"
