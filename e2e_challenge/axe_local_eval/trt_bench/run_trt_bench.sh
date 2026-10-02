#!/usr/bin/env bash
# Run the TensorRT speed benchmark for axe-v10 on one GPU.
#
#   CARD=4 e2e_challenge/axe_local_eval/trt_bench/run_trt_bench.sh
#
# Nothing here runs until a card is named: GPU use on this shared machine needs
# permission first, so CARD has no default and the script refuses without it.
#
# What it does: starts one axe-v10 container on that card -- the submitted
# image, unmodified -- with two host directories mounted read-only beside it:
#   trt_bench/   this benchmark's code
#   trt_site/    the TensorRT Python package, installed on the host rather than
#                baked into a new image. The disk was at 99% when this was
#                prepared and a 27 GB image build needs hundreds of GB of
#                transient snapshot space; a 3 GB site directory needs 3 GB.
# The TensorRT engine is built inside the container under /tmp, the same place
# the official environment would allow, and a copy of the result JSON comes out.
#
# It does not touch the driver, the simulator, or any other container.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/e2e_challenge/axe_local_eval/trt_bench"
SITE="${SITE:-$ROOT/../trt_site}"
IMG="${IMG:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10}"
CARD="${CARD:?name a GPU after getting permission, e.g. CARD=4}"
NEED_FREE="${NEED_FREE:-12000}"
OUT="$HERE/results"
NAME="axe-trt-bench-g$CARD"
mkdir -p "$OUT"

free="$(nvidia-smi -i "$CARD" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= NEED_FREE )) || { echo "GPU $CARD has only ${free} MiB free, need ${NEED_FREE}"; exit 2; }
[[ -d "$SITE/tensorrt_bindings" || -d "$SITE/tensorrt" ]] || { echo "TensorRT not installed at $SITE"; exit 2; }

stamp="$(date +%Y%m%d_%H%M%S)"
echo "[$(date '+%F %T')] benchmark on GPU $CARD (${free} MiB free) -> $OUT/bench_$stamp.json"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# The engine is written to a host directory so the driver-side runs can load the
# same plan instead of each driver rebuilding it on its first Drive call.
ENGINE_DIR="${ENGINE_DIR:-$HERE/engines}"
mkdir -p "$ENGINE_DIR"
docker run --rm --name "$NAME" --gpus "device=$CARD" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --tmpfs /tmp:rw,size=8g \
    -e PYTHONPATH="/trt_site:/bench" \
    -e LD_LIBRARY_PATH="/trt_site/tensorrt_libs:/trt_site/nvidia/cuda_runtime/lib" \
    -e TRT_WORKDIR=/engines \
    -v "$ENGINE_DIR:/engines" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$SITE:/trt_site:ro" \
    -v "$HERE:/bench:ro" \
    -v "$OUT:/out" \
    --entrypoint python "$IMG" \
    /bench/bench_trt.py --out "/out/bench_$stamp.json" "$@"
status=$?
echo "[$(date '+%F %T')] exit=$status"
[[ -f "$OUT/bench_$stamp.json" ]] && echo "result: $OUT/bench_$stamp.json"
exit $status
