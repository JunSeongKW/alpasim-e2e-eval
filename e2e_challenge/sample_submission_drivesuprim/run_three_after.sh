#!/usr/bin/env bash
# Waits for the 458-clip run to finish, then evaluates the three requested clips
# with video. Running both at once would put 19 renderers on the box, and the
# measured ceiling is 16 -- that stalls everything.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP=/tmp/claude-1000/-home-kaist5/f6b573e8-9153-4e4c-ba0a-564444b8bd06/scratchpad
RB=/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/e2e_challenge_drivesuprim_debug
while ! grep -q 'val sweep done' "$SP/from100ep.log" 2>/dev/null; do sleep 120; done
echo "=== 458 완료 확인, 3클립 영상 평가 시작 $(date +%H:%M:%S) ==="
sleep 60
P="$SCRIPT_DIR/assets/drivesuprim/stage3_ep24_eval"
IMAGE=alpasim-e2e-drivesuprim-stage3:from100ep \
EXPECTED_CHECKPOINT_SHA256=ee7c879b2cdbb7d6cbb5c8503cf2fea1070342f9bffd7b77f21ae30421e15e2d \
DRIVESUPRIM_BACKBONE_TYPE=bevformer_m \
DRIVESUPRIM_CONFIG_OVERRIDE="$P/stage3_config_refimi002.json" \
MPC_OVERRIDES="controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3" \
CLIPS_FILE="$SCRIPT_DIR/three_clips.txt" WORKERS=3 GPUS="1 2 3" \
BASE_PORT=6960 BASEPORT=6800 PREFIX=t3 RENDER_VIDEO=true \
SERVICE_STARTUP_TIMEOUT_SEC=5400 RUN_DIR="$RB/three_from100ep" \
    "$SCRIPT_DIR/run_val_sweep.sh"
echo "=== 3클립 완료 $(date +%H:%M:%S) ==="
