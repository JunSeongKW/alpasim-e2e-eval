#!/usr/bin/env bash
# Abort an arm if its drivers start swallowing inference errors.
#
# The driver catches any exception from the policy, logs it and keeps the
# previous plan, so a run whose reranker throws on every frame still produces
# ten finished rollouts that look ordinary and score like nothing happened.
# Two sweeps died that way before anyone read a driver log. This watches the
# logs while the arm drives and kills it the moment that starts.
set -uo pipefail
PREFIX="${1:?container prefix}"
LOG="${2:?arm log}"
LIMIT="${3:-20}"          # tolerate a few, abort on a pattern
while true; do
    sleep 60
    mapfile -t live < <(docker ps --filter "name=^${PREFIX}-g" --format '{{.Names}}' 2>/dev/null)
    [[ ${#live[@]} -eq 0 ]] && exit 0          # arm finished or was torn down
    total=0
    for c in "${live[@]}"; do
        n=$(docker logs "$c" 2>&1 | grep -c 'inference failed' || true)
        total=$((total + n))
    done
    if (( total > LIMIT )); then
        echo "[watchdog] $PREFIX: ${total} inference failures -- aborting arm" | tee -a "$LOG"
        docker logs "${live[0]}" 2>&1 | grep -A12 'inference failed' | tail -14 | tee -a "$LOG"
        pkill -f "run_10clips_reranker.sh" 2>/dev/null
        for p in $(ps -eo pid,args --no-headers | grep "alpasim_wizar[d].*${PREFIX#axe-rr-}" | awk '{print $1}'); do
            kill -9 "$p" 2>/dev/null
        done
        exit 1
    fi
done
