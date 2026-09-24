#!/usr/bin/env bash
# Steps 2 -> 3 -> 4 -> 5 of the gamma plan as ONE detached process.
#
# The first attempt at this chain lived inside the agent session's shell, and
# when that session ended at 11:54 the chain, the three validation arms, their
# wizards and their simulator stacks all went with it, 24 clips into 40. Only
# the driver containers survived, because docker does not care who started
# them. Everything that has to outlive a session runs under nohup from here on;
# this script is the chain, and it is launched with nohup.
#
#   nohup e2e_challenge/route_reranker/chain_validate_then_441.sh > .../chain.nohup 2>&1 &
#
# What it does:
#   ② resume the random-40 validation (RESUME=1: keep the run dirs, let the
#      runtime skip rollouts marked _complete; REUSE_DRIVERS=1: the 24 drivers
#      from the killed attempt are still up with the right gamma each)
#   ③ pick gamma by the paired rule in decide_gamma.py
#   ④ tear down the validation containers, run the 441 set at 16/16/16
#   ⑤ the 441 launcher prints the paired comparison against axe-v9 itself
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/chain.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ② validation: resume with the surviving drivers ==="
REUSE_DRIVERS=1 RESUME=1 GAMMAS="0 0.02 0.05" "$HERE/run_gamma_validate.sh"
log "② exit=$?"

for g in 0 0.02 0.05; do
    f="$ROOT/runs/val-g${g//./p}-cache-centre-max/aggregate/results-summary.json"
    [[ -f "$f" ]] || log "WARNING: gamma=$g has no summary"
done

log "=== ③ decide gamma ==="
"$ROOT/.venv/bin/python" "$HERE/decide_gamma.py" --explain 2>&1 | tee -a "$LOG"
G="$("$ROOT/.venv/bin/python" "$HERE/decide_gamma.py" 2>/dev/null | tail -1)"
if [[ -z "$G" ]]; then
    log "no gamma decided (validation incomplete?); falling back to 0.02"
    G=0.02
fi
log "chosen gamma: $G"

log "=== ④ clear validation containers, then 441 at 16/16/16 ==="
for c in $(docker ps -a --format '{{.Names}}' | grep -E '^(axe-rr-val|val-g)' | sort); do
    docker rm -f "$c" >/dev/null 2>&1 || true
done
log "validation containers left: $(docker ps -a --format '{{.Names}}' | grep -cE '^(axe-rr-val|val-g)')"

GAMMA="$G" "$HERE/run_441_reranker.sh" > "$HERE/441_g${G//./p}.nohup" 2>&1
log "④/⑤ 441 launcher exit=$? -- see $HERE/441_g${G//./p}.log"
log "=== chain done ==="
