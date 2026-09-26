#!/usr/bin/env bash
# Ten-minute progress for a run_441_reranker.sh run, written where it survives
# any session.  usage: progress_441.sh <tag>   e.g. g0p02-disjointep04
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim || exit 1
TAG="${1:?usage: progress_441.sh <tag>, e.g. g0p02-disjointep04}"
D="runs/rr441-${TAG}-cache-centre-max"
OUT="$D.progress.log"
LOG="e2e_challenge/route_reranker/441_${TAG}.log"
until grep -qE '^\[.*\] (OK:|FAILED:)' "$LOG" 2>/dev/null; do
  done_n=$(find "$D/rollouts" -name _complete 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] ${done_n}/441  gpu=$(timeout 10 nvidia-smi -i 0,1,2,3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd' ')  free=$(df -h / | tail -1 | awk '{print $4}')" >> "$OUT"
  sleep 600
done
echo "[$(date '+%F %T')] finished: $(grep -E '^\[.*\] (OK:|FAILED:)' "$LOG" | tail -1)" >> "$OUT"
