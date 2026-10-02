#!/usr/bin/env bash
# Sample per-process GPU memory of the TensorRT test's driver containers.
#
#   sample_vram.sh <out.csv> [interval_s]
#
# nvidia-smi reports host PIDs; each PID's cgroup names its container, which
# names the arm (axe-rr-trt8-fp32-* / axe-rr-trt8-trt-*). Only those
# containers' processes are written. Stops when no trt8 container is left.
set -uo pipefail
OUT="${1:?out.csv}"; EVERY="${2:-20}"
echo "ts,container,pid,used_mib" > "$OUT"
declare -A name_of
seen=0
while :; do
    ids="$(docker ps --filter 'name=^axe-rr-trt8-' --format '{{.ID}} {{.Names}}')"
    if [[ -z "$ids" ]]; then (( seen )) && break; sleep "$EVERY"; continue; fi
    seen=1
    while read -r id name; do name_of[$id]="$name"; done <<< "$ids"
    ts="$(date +%s)"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    while IFS=', ' read -r pid mib; do
        cid="$(grep -oE '[0-9a-f]{64}' "/proc/$pid/cgroup" 2>/dev/null | head -1)"
        [[ -n "$cid" ]] || continue
        n="${name_of[${cid:0:12}]:-}"
        [[ -n "$n" ]] && echo "$ts,$n,$pid,$mib" >> "$OUT"
    done
    sleep "$EVERY"
done
