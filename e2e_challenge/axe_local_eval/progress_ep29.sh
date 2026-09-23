#!/usr/bin/env bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
LOG=e2e_challenge/axe_local_eval/260923_ep29_attempt5.nohup.log
RD=runs/leaderboard-260923-ep29-step30330
OUT=/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/leaderboard-260923-ep29-step30330.progress.log
t0=$(date -d '2026-09-23 19:23:52' +%s)
while ! grep -q 'wizard exit' "$LOG" 2>/dev/null; do
  n=$(find "$RD/rollouts" -name 'rollout.asl' 2>/dev/null | wc -l)
  echo "[$(date '+%T') +$(( ($(date +%s)-t0)/60 ))min] rollout.asl: $n/441  gpu: $(timeout 10 nvidia-smi -i 0,1,2,3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd' ')  free: $(df -h /home | tail -1 | awk '{print $4}')" >> "$OUT"
  sleep 600
done
echo "=== launcher finished $(date '+%T') ===" >> "$OUT"; tail -4 "$LOG" | cut -c1-160 >> "$OUT"
ls "$RD/aggregate/" >> "$OUT" 2>&1
