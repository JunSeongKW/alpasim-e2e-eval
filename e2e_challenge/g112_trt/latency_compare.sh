#!/usr/bin/env bash
# Inference time of axe-v9, vits512_g112_kv28 (PyTorch) and the g112 TensorRT
# image under the official ec2 preset, then each model alone on one GPU.
#
#   nohup e2e_challenge/g112_trt/latency_compare.sh > runs/g112-trt-latency-20261013.progress.log 2>&1 &
#
# 1. ec2 preset as run_ec2_timing.py runs it (8gpu_32rollouts: 16 driver replicas
#    x 2 concurrent rollouts on GPUs 4-7, 32 renderers on 0-3, harmonizer off,
#    BATCH_RENDER_RGB, no driver call during forced GT, gains 1.0/0.25/3), the same
#    first SCENES curated_val scenes for each model, one model after another, each
#    waiting until all eight GPUs are under 4 GB. 96 scenes = three waves of the 32
#    slots (~17k Drive calls per model), enough for the latency percentiles and a
#    steady-rate estimate of the 441 wall time against the 14,400 s budget.
# 2. Model alone (bench_models.py: DriveSuprimPolicy._run_agent, batch 1, 30 warm-up
#    + 200 timed calls), the three side by side on GPUs 4-6 once those are free.
# 3. One table: runs/ec2-timing-summary-<DATE>.md. The TensorRT drivers must all
#    report the exact engine active.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
DATE="${DATE:-20261013}"
SCENES="${SCENES:-96}"
MODELS=(axe-v9 vits512_g112_kv28 vits512_g112_kv28_trt)
PY=.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

log "$(python3 "$HERE/register_trt_model.py")"
for model in "${MODELS[@]}"; do
    run="ec2-timing-${model//_/}-${DATE}-first${SCENES}"
    if [[ -f "runs/$run/aggregate/results-summary.json" ]]; then log "$model already done"; continue; fi
    log "$model: queued ($SCENES scenes, ec2 preset; waits for GPUs 0-7 under 4 GB)"
    $PY e2e_challenge/pkg3cam_eval/run_ec2_timing.py --model "$model" --date "$DATE" --scenes "$SCENES" \
        --wait >> "runs/$run.progress.log" 2>&1 && log "$model: done" || log "$model: FAILED (runs/$run.progress.log)"
done

L="runs/ec2-timing-vits512g112kv28trt-${DATE}-first${SCENES}/driver-logs"
n=$(ls "$L"/*.log 2>/dev/null | wc -l)
active=$(grep -l "TRT backbone: active engine=exact" "$L"/*.log 2>/dev/null | wc -l)
calls=$(cat "$L"/*.log 2>/dev/null | grep -cE "TRT backbone: call [0-9]+" || true)
log "TensorRT drivers: $active / $n with the exact engine active; fallback or shape-miss lines: $calls"

log "model alone: waiting for GPUs 4-6 under 4 GB"
until nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' '$1>=4 && $1<=6 && $2>=4000 {busy=1} END {exit busy}'; do sleep 60; done
out="$($PY e2e_challenge/pkg3cam_eval/run_bench_models.py --gpus 4,5,6 \
        --models "$(IFS=,; echo "${MODELS[*]}")" | tee /dev/stderr | sed -n 's/^results: //p')"
grep -h "TRT backbone" "$out"/vits512_g112_kv28_trt.log | head -2

$PY e2e_challenge/pkg3cam_eval/run_ec2_timing.py --table --date "$DATE" --scenes "$SCENES" \
    --models "$(IFS=,; echo "${MODELS[*]}")" ${out:+--bench "$out"}
log "table: runs/ec2-timing-summary-${DATE}.md"
