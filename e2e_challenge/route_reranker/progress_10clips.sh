#!/usr/bin/env bash
# Progress of one 10-clip arm, appended every 5 min to runs/rerank10-<arm>.progress.log
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
ARM="${1:?before|rerank}"; TAG="${RUN_TAG:-rerank10}"
LOG=e2e_challenge/route_reranker/10clips_${TAG}_${ARM}.log
OUT=runs/$TAG-$ARM.progress.log
while ! grep -q 'wizard exit' "$LOG" 2>/dev/null; do
  echo "[$(date '+%T')] asl=$(find runs/$TAG-$ARM/rollouts -name 'rollout.asl' 2>/dev/null | wc -l)/10 mp4=$(find runs/$TAG-$ARM -name '*.mp4' 2>/dev/null | wc -l) gpu=$(timeout 10 nvidia-smi -i 0,1,2,3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd' ')" >> "$OUT"
  sleep 300
done
echo "[$(date '+%T')] finished: $(tail -1 "$LOG")" >> "$OUT"
