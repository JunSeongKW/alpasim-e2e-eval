#!/usr/bin/env bash
# Delete a runs/.trash-*/ directory once no evaluation of ours is running.
#
#   nohup tools/delete_trash_when_idle.sh runs/.trash-cleanup-20261009 > runs/cleanup-20261009.progress.log 2>&1 &
#
# Why wait: deleting tens of GB on this ext4 volume stalls the journal and has
# killed a running evaluation before (AGENTS.md), so cleanup is a rename into
# .trash-*/ first and the real delete happens here, later. "Idle" means no
# runs/*/status.json in starting_drivers / simulation, no finalize_after_eval.sh
# (fit + combined table) and no axe-* or drivor-* compose containers, seen twice
# ten minutes apart. The delete itself is
# nice/ionice idle and goes one file at a time with a pause, so other users'
# jobs on the same disk are not starved either.
set -uo pipefail
TRASH="${1:?trash directory, e.g. runs/.trash-cleanup-20261009}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "$(basename "$TRASH")" in .trash-*) ;; *) echo "refusing: $TRASH is not a .trash-* directory"; exit 2;; esac
[[ -d "$TRASH" ]] || { echo "no such directory: $TRASH"; exit 2; }

busy() {
    for s in runs/*/status.json; do
        stage=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('stage',''))" "$s" 2>/dev/null)
        [[ "$stage" == simulation || "$stage" == starting_drivers ]] && { echo "$s:$stage"; return 0; }
    done
    ps -eo cmd | grep -q "[f]inalize_after_eval.sh" && { echo "finalize running"; return 0; }
    n=$(docker ps --format '{{.Names}}' | grep -cE '^(axe-|drivor-)')
    [[ "$n" -gt 0 ]] && { echo "containers:$n"; return 0; }
    return 1
}

echo "[$(date '+%F %T')] waiting until no evaluation runs ($(du -sh --apparent-size "$TRASH" | cut -f1) in $TRASH)"
quiet=0
while [[ $quiet -lt 2 ]]; do
    if why=$(busy); then quiet=0; else quiet=$((quiet + 1)); why=idle; fi
    [[ $quiet -lt 2 ]] && sleep 600
done
echo "[$(date '+%F %T')] idle; deleting slowly"
n=0
while IFS= read -r -d '' f; do
    ionice -c3 nice -n19 rm -f -- "$f"
    n=$((n + 1))
    (( n % 50 == 0 )) && echo "[$(date '+%F %T')] $n files removed"
    sleep 0.3
done < <(find "$TRASH" -type f -size +10M -print0)
ionice -c3 nice -n19 rm -rf -- "$TRASH"
echo "[$(date '+%F %T')] done: removed $TRASH ($n large files); free now $(df -h --output=avail "$ROOT" | tail -1)"
