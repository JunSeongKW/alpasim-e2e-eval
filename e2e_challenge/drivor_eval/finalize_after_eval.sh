#!/usr/bin/env bash
# After both DrivoR 441x3 runs finish: Drive-IRT fit with the 8 references, the
# three scoring conventions (challenge / Alpamayo / RoaD, rescore_protocols.py),
# then the combined AlpaSim 441 report (our runs + organizer references + published rows).
#
#   nohup e2e_challenge/drivor_eval/finalize_after_eval.sh 20261009 > runs/drivor-finalize-20261009.progress.log 2>&1 &
#
# Waits while either run is simulating or paused; stops if one failed. The fit
# refuses runs with driver errors or infrastructure (RPC) failures -- then the
# table is still written, without PCS, and the reason is in the log.
set -uo pipefail
DATE="${1:?date of the runs, e.g. 20261009}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ORIG="runs/drivor-drivor-original-441x3-${DATE}"
OURS="runs/drivor-ours-hugsim-final-441x3-${DATE}"
FIT="runs/leaderboard-drivor-8ref-${DATE}"
OUT="runs/drivor-combined-alpasim441-${DATE}"
stage() { python3 -c "import json;print(json.load(open('$1/status.json'))['stage'])" 2>/dev/null || echo missing; }
echo "[$(date '+%F %T')] waiting for $ORIG and $OURS"
while :; do
    a=$(stage "$ORIG"); b=$(stage "$OURS")
    case "$a$b" in *failed*) echo "[$(date '+%F %T')] a run failed: original=$a ours=$b"; exit 1;; esac
    if [[ "$a" == complete* && "$b" == complete* ]]; then break; fi
    sleep 300
done
echo "[$(date '+%F %T')] both complete (original=$a ours=$b); fitting"
if .venv/bin/python e2e_challenge/drivor_eval/fit_drivor.py --original "$ORIG" --ours "$OURS" --out "$FIT"; then
    echo "[$(date '+%F %T')] fit done: $FIT/drivor_leaderboard.csv"
else
    echo "[$(date '+%F %T')] fit refused (see above); the table is written without PCS"
fi
PROT="runs/drivor-protocols-${DATE}"
UV_OFFLINE=1 .venv/bin/python e2e_challenge/drivor_eval/rescore_protocols.py \
    --run ours="$OURS" --run original="$ORIG" \
    --check-against ours="$OURS/aggregate/results-summary.json" \
    --check-against original="$ORIG/aggregate/results-summary.json" --out "$PROT" \
    && echo "[$(date '+%F %T')] protocols: $PROT/protocols.md" \
    || echo "[$(date '+%F %T')] protocol rescoring failed; table without it"
PROTARG=(); [[ -f "$PROT/protocols.json" ]] && PROTARG=(--protocols "$PROT/protocols.json")
.venv/bin/python e2e_challenge/drivor_eval/combined_table.py --original "$ORIG" --ours "$OURS" --fit "$FIT" --out "$OUT" "${PROTARG[@]}"
echo "[$(date '+%F %T')] table: $OUT/combined_alpasim441.md"
