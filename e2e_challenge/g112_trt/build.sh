#!/usr/bin/env bash
# Build the vits512_g112_kv28 TensorRT submission image (see Dockerfile).
#
#   GPU=0 e2e_challenge/g112_trt/build.sh            # engines (once, on that GPU) + image
#
# Everything comes from what the 441 evaluation of this model ran:
#   bundle   ../models/stage3_vits512_g112_kv28_best_20261010 (checksums verified)
#   adapter  runs/prepare-stage3-vits512-g112-kv28-20261010/drivesuprim_challenge
#   config   runs/prepare-stage3-vits512-g112-kv28-20261010/assets/config.json
# plus trt_backbone_runtime.py and a two-line hook at the end of
# DriveSuprimPolicy.__init__ that calls it. Engines are built once into
# runs/g112-trt-build/engines and reused; the image build itself is offline.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUNDLE="$ROOT/../models/stage3_vits512_g112_kv28_best_20261010"
SRC="$BUNDLE/AXEv1.0-NuRec"
PREP="$ROOT/runs/prepare-stage3-vits512-g112-kv28-20261010"
SITE="$ROOT/../trt_site"
BASE='696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765'
SHA=f048fe9c5bc1aaf069d92a023cacb668f2c5e74b2496f107b26e75d347357843
TAG="${TAG:-alpasim-e2e-drivesuprim-stage3:g112-kv28-trt}"
OUT="$ROOT/runs/g112-trt-build"
GPU="${GPU:-0}"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "1. inputs"
(cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS)
CKPT="$SRC/checkpoints/stage3_best.ckpt"
[[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" == "$SHA" ]] || { log "checkpoint sha mismatch"; exit 2; }
python3 - "$PREP" "$SHA" <<'EOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/model.json"))
assert m["checkpoint_sha256"] == sys.argv[2] and m["pinhole_profile"] == "K377_512X256", m
assert json.load(open(f"{sys.argv[1]}/smoke/gpu_validation.json"))["passed"]
EOF
docker image inspect "$BASE" >/dev/null
mkdir -p "$OUT/engines" && chmod 777 "$OUT/engines"

if [[ ! -f "$OUT/engines/manifest.json" ]]; then
    log "2. engines on GPU $GPU"
    docker run --rm --name g112-trt-build-engines --gpus "device=$GPU" --network none \
        --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python \
        -v "$SRC/navsim:/app/navsim:ro" -v "$SRC/nurec_pf:/app/nurec_pf:ro" \
        -v "$PREP/drivesuprim_challenge:/app/drivesuprim_challenge:ro" \
        -v "$CKPT:/app/assets/drivesuprim/pkg_stage3_best.ckpt:ro" \
        -v "$PREP/assets/config.json:/app/assets/drivesuprim/pkg_config.json:ro" \
        -e DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/pkg_stage3_best.ckpt \
        -e DRIVESUPRIM_CONFIG_PATH=/app/assets/drivesuprim/pkg_config.json \
        -e DRIVESUPRIM_PINHOLE_PROFILE=K377_512X256 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/trt_site \
        -v "$SITE:/trt_site:ro" -v "$HERE:/g112_trt:ro" \
        -v "$ROOT/e2e_challenge/axe_local_eval/trt_bench:/checks:ro" -v "$OUT/engines:/out" \
        "$BASE" /g112_trt/build_engines.py 2>&1 | tee "$OUT/build_engines.log"
    [[ -f "$OUT/engines/manifest.json" ]] || { log "engine build failed"; exit 1; }
fi

log "3. build context"
mkdir -p "$ROOT/.cache"
STAGE="$(mktemp -d "$ROOT/.cache/g112-trt-build.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
A="$STAGE/app"; S="$STAGE/site"
mkdir -p "$A/assets/drivesuprim/trt" "$S/tensorrt_libs" "$S/nvidia"
cp -a "$SRC/navsim" "$SRC/nurec_pf" "$A/"
cp -a "$PREP/drivesuprim_challenge" "$A/"
cp "$HERE/trt_backbone_runtime.py" "$A/drivesuprim_challenge/"
python3 - "$A/drivesuprim_challenge/policy.py" <<'EOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
anchor = "        self._bev_debugger = BevDebugger(self._agent, self._config)\n"
assert s.count(anchor) == 1, "hook anchor not found exactly once"
hook = ("\n        # TensorRT image backbone: installed only after it matches the PyTorch\n"
        "        # backbone on this GPU; never raises (trt_backbone_runtime.py).\n"
        "        from .trt_backbone_runtime import install as _install_trt_backbone\n\n"
        "        self._trt_engine = _install_trt_backbone(self)\n")
p.write_text(s.replace(anchor, anchor + hook))
EOF
cp "$CKPT" "$A/assets/drivesuprim/pkg_stage3_best.ckpt"
cp "$PREP/assets/config.json" "$A/assets/drivesuprim/pkg_config.json"
cp "$OUT/engines/manifest.json" "$OUT"/engines/backbone_*_fp16.plan "$A/assets/drivesuprim/trt/"
# TensorRT runtime only: the bindings, libnvinfer + plugin + ONNX parser (the
# bindings' tensorrt.so links all three, so `import` fails without any of them),
# and the CUDA 11.8 runtime the libs package loads beside itself. Only the
# builder resources (1.5 GB + a Windows copy) are build-time only.
cp -a "$SITE/tensorrt_bindings" "$SITE"/tensorrt_cu11_bindings-*.dist-info "$SITE"/tensorrt_cu11_libs-*.dist-info "$S/"
cp "$SITE/tensorrt_libs/__init__.py" "$SITE/tensorrt_libs/libnvinfer.so.10" "$SITE/tensorrt_libs/libnvinfer_plugin.so.10" \
   "$SITE/tensorrt_libs/libnvonnxparser.so.10" "$S/tensorrt_libs/"
cp -a "$SITE/nvidia/__init__.py" "$SITE/nvidia/cuda_runtime" "$S/nvidia/"
cp -a "$SITE"/nvidia_cuda_runtime_cu11-*.dist-info "$S/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$HERE/Dockerfile" "$STAGE/"
du -sh "$A" "$S"

log "4. docker build $TAG"
docker build --network none \
    --build-arg "CHECKPOINT_SHA256=$SHA" \
    --build-arg "SOURCE_MANIFEST_SHA256=$(sha256sum "$BUNDLE/SHA256SUMS" | cut -d' ' -f1)" \
    --build-arg "CONFIG_SHA256=$(sha256sum "$PREP/assets/config.json" | cut -d' ' -f1)" \
    --build-arg "ENGINES_MANIFEST_SHA256=$(sha256sum "$OUT/engines/manifest.json" | cut -d' ' -f1)" \
    -t "$TAG" "$STAGE" 2>&1 | tail -5
docker image inspect "$TAG" --format '{{.Id}} {{.Size}}' | tee "$OUT/image.txt"
log "done: $TAG"
