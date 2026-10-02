#!/usr/bin/env bash
# One arm of the TensorRT speed test: axe-v10 on the eight clips of clips8.txt.
#
#   KIND=fp32 GPUS=0,1,2,3 e2e_challenge/axe_local_eval/trt_bench/run_trt_8clips.sh
#   KIND=trt  GPUS=4,5,6,7 TRT_PLAN=backbone_fp16_<digest>.plan \
#             TRT_TARGET=<module path> e2e_challenge/axe_local_eval/trt_bench/run_trt_8clips.sh
#
# Both arms are the axe-v10 configuration exactly (image, checkpoint, reranker
# cache-centre-max at gamma 0.01, dev preset, MPC gains) and both log the model
# time of every Drive call (DRIVESUPRIM_TIMING=1). The only difference is that
# the trt arm mounts the TensorRT runtime and a prebuilt engine and swaps the
# image backbone for it when the driver starts.
#
# Shape: 8 clips on 4 cards, 2 drivers and 2 renderers per card, one wave. That
# is half the official density (4 drivers per card), stated in the report; both
# arms use the same shape, so the comparison between them is fair.
#
# The goal is speed, not driving score: video is off and one rollout per clip.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/e2e_challenge/axe_local_eval/trt_bench"
SITE="${SITE:-$ROOT/../trt_site}"
KIND="${KIND:?KIND=fp32|trt}"
GPUS="${GPUS:?GPUS=0,1,2,3 or 4,5,6,7}"
STAMP="${STAMP:-$(date +%m%d_%H%M)}"

extra="-e DRIVESUPRIM_TIMING=1"
case "$KIND" in
    fp32) BASE_PORT="${BASE_PORT:-8400}"; WIZARD_BASEPORT="${WIZARD_BASEPORT:-22000}" ;;
    trt)
        TRT_PLAN="${TRT_PLAN:?name the plan file under trt_bench/engines}"
        TRT_TARGET="${TRT_TARGET:?module path of the converted backbone}"
        [[ -f "$HERE/engines/$TRT_PLAN" ]] || { echo "missing $HERE/engines/$TRT_PLAN"; exit 2; }
        [[ -d "$SITE/tensorrt_bindings" ]] || { echo "TensorRT not installed at $SITE"; exit 2; }
        BASE_PORT="${BASE_PORT:-8500}"; WIZARD_BASEPORT="${WIZARD_BASEPORT:-23000}"
        # LD_LIBRARY_PATH keeps the image's own entries after TensorRT's.
        extra+=" -v $SITE:/trt_site:ro -v $HERE:/bench:ro -v $HERE/engines:/engines:ro"
        extra+=" -e PYTHONPATH=/trt_site:/bench"
        extra+=" -e LD_LIBRARY_PATH=/trt_site/tensorrt_libs:/trt_site/nvidia/cuda_runtime/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
        extra+=" -e DRIVESUPRIM_TRT_PLAN=/engines/$TRT_PLAN -e DRIVESUPRIM_TRT_TARGET=$TRT_TARGET"
        ;;
    *) echo "unknown KIND=$KIND"; exit 2 ;;
esac

NEED_FREE="${NEED_FREE:-30000}"
IFS=',' read -r -a cards <<< "$GPUS"
for c in "${cards[@]}"; do
    free="$(nvidia-smi -i "$c" --query-gpu=memory.free --format=csv,noheader,nounits)"
    (( free >= NEED_FREE )) || { echo "GPU $c has only ${free} MiB free, need ${NEED_FREE}"; exit 2; }
done

RUN_TAG="trt8-$KIND-$STAMP"
PROGRESS="$ROOT/runs/$RUN_TAG.progress.log"
mkdir -p "$ROOT/runs"
git -C "$ROOT" rev-parse HEAD > "$ROOT/runs/$RUN_TAG.git-commit.txt" 2>/dev/null || true
echo "[$(date '+%F %T')] $RUN_TAG on GPUs $GPUS, ports $BASE_PORT/$WIZARD_BASEPORT" | tee -a "$PROGRESS"

t0=$(date +%s)
ARM=cache-centre-max ROUTE_RERANK_WEIGHT=0.01 RUN_TAG="$RUN_TAG" \
IMG=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \
EXPECTED_CHECKPOINT_SHA256=29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8 \
CLIPS="$HERE/clips8.txt" GPUS="$GPUS" RENDER_GPUS="$GPUS" REPLICAS=2 ROLLOUT_WORKERS=8 \
RENDER_VIDEO=false WATCH_LIMIT=20 \
BASE_PORT="$BASE_PORT" WIZARD_BASEPORT="$WIZARD_BASEPORT" \
EXTRA_DOCKER_ARGS="$extra" \
    "$ROOT/e2e_challenge/route_reranker/run_10clips_reranker.sh" >> "$PROGRESS" 2>&1
status=$?
echo "[$(date '+%F %T')] $RUN_TAG exit=$status wall_s=$(( $(date +%s) - t0 ))" | tee -a "$PROGRESS"
exit $status
