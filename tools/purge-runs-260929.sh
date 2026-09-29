#!/usr/bin/env bash
# Reduce runs/ to what a future leaderboard fit or per-clip analysis actually reads.
#
# The rule, not a list of exceptions: `evaluate.py` and `per_clip_csv.py` open
# exactly one file per run, `aggregate/results-summary.json`. Everything else in a
# run directory is either an input to that file (already consumed) or a by-product.
# So the keep-set is small and explicit, and everything under runs/ that is not in
# it goes.
#
# KEPT
#   */aggregate/results-summary.json   the only file a leaderboard fit reads
#   *provenance*.json, */git-commit.txt  which checkpoint and settings produced it
#   the leaderboard fit outputs (capability_ranking.csv and friends)
#   runs/*.jsonl                       the gamma ledger
#   runs/clip1-ep29-*/**/*.mp4         the three single-clip videos just delivered,
#                                      which are in active use (they were renamed
#                                      by hand to g0.01 / g0.02 / norerank)
#
# DROPPED (~63 GB)
#   59.2 GB  rollout.asl               the raw sensor log; scoring is done
#    3.5 GB  *.wizard.log, *.driver-*.log
#    0.8 GB  every other .mp4          24 videos from finished experiments
#    0.2 GB  *.parquet                 per-rollout metrics, already aggregated
#    0.1 GB  telemetry/, *.png, txt-logs/, controller/
#
# Two things are done before deleting, so no measurement is lost with the data:
#   * the stopped gamma=0.005 run (129/441, no summary) is scored first, and its
#     partial summary written into its aggregate/
#   * the driver logs' FEASGATE/RERANK statistics -- what the collision-gate
#     analysis was built on -- are extracted to runs/driver_log_stats.json
#
# Throttled: one ext4 filesystem, other people's jobs writing to it. When the
# gamma run was killed, `docker rm -f` hung for minutes because jbd2 was blocked.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/runs/purge-260929.log"
PY="$ROOT/.venv/bin/python"
BATCH="${BATCH:-12}"
PAUSE="${PAUSE:-2}"
DRY="${DRY:-0}"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

G005="runs/rr441-g0p005-augep29-cache-centre-max"

# 1. Score the stopped run before its rollouts go.
if [[ -d "$G005/rollouts" && ! -f "$G005/aggregate/results-summary.json" ]]; then
    log "scoring the stopped gamma=0.005 run ($(find "$G005/rollouts" -name _complete | wc -l) finished rollouts)"
    if "$PY" e2e_challenge/axe_local_eval/interim_score.py "$G005" >>"$LOG" 2>&1; then
        src="/tmp/interim-score/0-$(basename "$G005")/results-summary.json"
        if [[ -f "$src" ]]; then
            mkdir -p "$G005/aggregate"
            cp "$src" "$G005/aggregate/results-summary.json"
            log "  saved: $("$PY" -c "
import json
rs=[r for r in json.load(open('$src'))['rollouts'] if r.get('score') is not None]
print(f'{len(rs)} clips, mean {sum(r[\"score\"] for r in rs)/len(rs):.4f}')")"
        else
            log "  ABORT: interim scoring produced no summary"; exit 2
        fi
    else
        log "  ABORT: interim scoring failed; keeping everything"; exit 2
    fi
fi

# 2. Extract what the gate analysis used, then the logs are expendable.
if compgen -G "runs/*.driver-*.log" >/dev/null; then
    log "extracting FEASGATE/RERANK statistics from driver logs"
    "$PY" - >>"$LOG" 2>&1 <<'EOF'
import collections, glob, json, os, re, statistics
runs = collections.defaultdict(list)
for p in glob.glob("runs/*.driver-*.log"):
    runs[re.sub(r"\.driver-[0-9a-f]+\.log$", "", os.path.basename(p))].append(p)
out = {}
for run, paths in sorted(runs.items()):
    co, dr, rerank, changed = [], [], 0, 0
    for p in paths:
        with open(p, errors="replace") as f:
            for line in f:
                if "FEASGATE" in line:
                    m = re.search(r"by_drivable=(\d+)", line)
                    n = re.search(r"by_collision=(\d+)", line)
                    if m: dr.append(int(m.group(1)))
                    if n: co.append(int(n.group(1)))
                elif "RERANK:" in line:
                    rerank += 1
                    g = re.search(r"orig=#(\d+).*sel=#(\d+)", line)
                    if g and g.group(1) != g.group(2):
                        changed += 1
    rec = {"drivers": len(paths), "feasgate_frames": len(co),
           "rerank_frames": rerank, "rerank_changed": changed}
    if co:
        rec["collision_gate_fired_frac"] = round(sum(1 for x in co if x > 0) / len(co), 4)
        rec["collision_dropped_mean"] = round(statistics.fmean(co), 1)
    if dr:
        rec["drivable_gate_fired_frac"] = round(sum(1 for x in dr if x > 0) / len(dr), 4)
        rec["drivable_dropped_mean"] = round(statistics.fmean(dr), 1)
    out[run] = rec
    print(f"  {run}: gate frames {len(co)}, rerank {rerank}, changed {changed}")
json.dump(out, open("runs/driver_log_stats.json", "w"), indent=1)
print(f"  -> runs/driver_log_stats.json ({len(out)} runs)")
EOF
fi

# 3. Build the keep-set.
keep="$(mktemp)"; trap 'rm -f "$keep"' EXIT
{
    find runs -name 'results-summary.json'
    find runs -name '*provenance*.json' -o -name 'git-commit.txt'
    find runs/leaderboard-260925 runs/leaderboard-260925b runs/leaderboard-260925c \
         runs/leaderboard-260926 runs/leaderboard-260927 runs/leaderboard-260928 \
         runs/pool-sensitivity -type f 2>/dev/null
    find runs -maxdepth 1 -name '*.jsonl'
    find runs -maxdepth 1 -name 'driver_log_stats.json'
    find runs -maxdepth 1 -name 'purge-*.log' -o -maxdepth 1 -name 'purge-*.txt'
    find runs/clip1-ep29-norerank-before runs/clip1-ep29-g0p01-cache-centre-max \
         runs/clip1-ep29-g0p02-cache-centre-max -name '*.mp4' 2>/dev/null
} | sort -u > "$keep"
log "keep-set: $(wc -l < "$keep") files"

all="$(mktemp)"; find runs -type f | sort -u > "$all"
drop="$(mktemp)"; comm -23 "$all" "$keep" > "$drop"
bytes="$(xargs -r -d '\n' stat -c %s < "$drop" 2>/dev/null | awk '{s+=$1} END {print s+0}')"
log "to delete: $(wc -l < "$drop") files, $(( bytes / 1024 / 1024 / 1024 )) GB"

if [[ "$DRY" == 1 ]]; then
    log "DRY=1: nothing deleted. Largest to go:"
    xargs -r -d '\n' stat -c '%s %n' < "$drop" 2>/dev/null | sort -rn | head -10 \
        | awk '{printf "  %8.1f MB  %s\n", $1/1e6, $2}' | tee -a "$LOG"
    rm -f "$all" "$drop"; exit 0
fi

before_kb="$(df --output=avail -k / | tail -1)"
n=0
while IFS= read -r f; do
    ionice -c 3 -t nice -n 19 rm -f "$f"
    n=$((n + 1))
    if (( n % BATCH == 0 )); then
        sleep "$PAUSE"
        (( n % 600 == 0 )) && log "  ${n} files"
    fi
done < "$drop"
find runs -depth -type d -empty -delete 2>/dev/null || true
rm -f "$all" "$drop"

after_kb="$(df --output=avail -k / | tail -1)"
log "freed $(( (after_kb - before_kb) / 1024 / 1024 )) GB ; $n files removed"
log "runs/ is now $(du -sh runs | cut -f1)"
log "kept $(find runs -name 'results-summary.json' | wc -l) summaries, $(find runs -name '*.mp4' | wc -l) videos"
