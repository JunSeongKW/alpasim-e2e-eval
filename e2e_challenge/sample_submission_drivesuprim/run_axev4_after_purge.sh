#!/usr/bin/env bash
# Wait for the container purge to finish, then evaluate axe-v4 over the 458
# clips. Running both at once races: the purge removes stopped containers, and
# the eval's containers are briefly "created" before they run.
set -u
SP=/tmp/claude-1000/-home-kaist5/f6b573e8-9153-4e4c-ba0a-564444b8bd06/scratchpad
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RB=/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/e2e_challenge_drivesuprim_debug

while ! grep -q '정리 완료' "$SP/purge.log" 2>/dev/null; do sleep 60; done
echo "=== 컨테이너 정리 완료, axe-v4 평가 시작 $(date +%H:%M:%S) ==="
sleep 30

rm -rf "$RB/val458_axev4"
IMAGE=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v4 \
CLIPS_FILE="$SD/nurec_val_clips_ids.txt" \
RUN_DIR="$RB/val458_axev4" \
WORKERS=16 GPUS="0 1 2 3" BASE_PORT=6900 BASEPORT=6400 PREFIX=av4 \
RENDER_VIDEO=false SERVICE_STARTUP_TIMEOUT_SEC=5400 \
    "$SD/run_axev5_sweep.sh"
echo "=== axe-v4 완료 $(date +%H:%M:%S) ==="
