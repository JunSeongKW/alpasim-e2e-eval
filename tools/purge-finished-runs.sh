#!/usr/bin/env bash
# Free disk by removing the raw rollouts of finished experiments, throttled so a
# concurrent evaluation is not disturbed.
#
# What goes: rollouts/, txt-logs/ and controller/ under runs/{gen,val,gam,rerank10}-*
# -- the 38-clip generalisation sweep, the 40-clip validation, the 1-clip gamma
# sweep and the 10-clip reranker pair. All four are finished and every number
# that was reported from them comes out of aggregate/results-summary.json, which
# stays, together with aggregate/ (per-rollout metrics), telemetry/, the configs
# and the summary videos. This is the same sweep the 441 launchers already do at
# the end of a run, applied after the fact to the runs that predate it.
#
# What stays: every runs/leaderboard-* summary, every aggregate/, the reranker
# 10-clip videos the user asked for, .cache/renderer-shared.
#
# Why throttled. Everything here is one ext4 filesystem (/dev/vda1), and a
# researcher's 16-renderer evaluation has been writing to it for three days.
# Freeing 130 GB in one rm floods the journal and stalls whoever else is writing
# -- the rule in AGENTS.md exists because that has happened. So: idle I/O class,
# one large file at a time, a pause between batches, and a health check on the
# other stack in between that aborts rather than risk it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/runs/purge-260926.log"
BATCH="${BATCH:-15}"        # files per batch
PAUSE="${PAUSE:-2}"         # seconds between batches
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# The other evaluation, by name only. Never act on it -- just refuse to keep
# deleting if it goes away while we are working.
OTHER_PREFIX='axe-ep30n-'
other_count() { docker ps --format '{{.Names}}' | grep -c "^$OTHER_PREFIX" || true; }
BASE_OTHER="$(other_count)"
log "other evaluation ($OTHER_PREFIX*): $BASE_OTHER containers up at start"

before_kb="$(df --output=avail -k / | tail -1)"

# Collect targets. Only these four run families, only these three subdirectories.
targets=()
for d in runs/gen-*-cache-centre-max runs/val-*-cache-centre-max \
         runs/gam-*-cache-centre-max runs/rerank10-before runs/rerank10-rerank; do
    [[ -d "$d" ]] || continue
    for sub in rollouts txt-logs controller; do
        [[ -d "$d/$sub" ]] && targets+=("$d/$sub")
    done
done
targets+=(".trash-260923")
log "targets: ${#targets[@]} directories"
printf '  %s\n' "${targets[@]}" >> "$LOG"

n=0
for t in "${targets[@]}"; do
    [[ -e "$t" ]] || continue
    sz="$(du -sm "$t" 2>/dev/null | cut -f1)"
    log "--- $t (${sz} MB)"
    # One file at a time, in batches, so the freed space appears early and the
    # journal sees the work spread out rather than all at once.
    while IFS= read -r -d '' f; do
        ionice -c 3 -t nice -n 19 rm -f "$f"
        n=$((n + 1))
        if (( n % BATCH == 0 )); then
            now="$(other_count)"
            if (( now < BASE_OTHER )); then
                log "ABORT: other evaluation dropped from $BASE_OTHER to $now containers"
                exit 3
            fi
            sleep "$PAUSE"
        fi
    done < <(find "$t" -type f -print0 2>/dev/null)
    # Whatever is left is empty directories and symlinks: cheap.
    ionice -c 3 -t nice -n 19 rm -rf "$t"
    log "    removed ($n files so far)"
done

after_kb="$(df --output=avail -k / | tail -1)"
log "freed $(( (after_kb - before_kb) / 1024 / 1024 )) GB ; $n files removed"
log "other evaluation still up: $(other_count) containers (was $BASE_OTHER)"
