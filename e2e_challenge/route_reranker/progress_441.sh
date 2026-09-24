#!/usr/bin/env bash
# Ten-minute progress line for a 441 reranker run, into the run directory so it
# outlives any agent session.  usage: progress_441.sh <tag>   e.g. g0p02
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
TAG="${1:?tag, e.g. g0p02}"
D="runs/rr441-$TAG-cache-centre-max"
OUT="$D.progress.log"
LOG="e2e_challenge/route_reranker/441_$TAG.log"
until grep -qE '^\[.*\] (OK|FAILED):' "$LOG" 2>/dev/null; do
    echo "[$(date '+%T')] $(find "$D/rollouts" -name _complete 2>/dev/null | wc -l)/441" \
         " gpu=$(timeout 10 nvidia-smi -i 0,1,2,3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd' ')" \
         " free=$(df -h /home | tail -1 | awk '{print $4}')" >> "$OUT"
    sleep 600
done
echo "[$(date '+%T')] finished: $(grep -E '^\[.*\] (OK|FAILED):' "$LOG" | tail -1)" >> "$OUT"
