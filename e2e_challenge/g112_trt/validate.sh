#!/usr/bin/env bash
# Everything that can go wrong with the g112 TensorRT image, checked locally before any quota.
#
#   GPU=4 e2e_challenge/g112_trt/validate.sh [IMAGE]
#   ECR_URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:<tag> ... (adds check 10)
#
#  1 image, tag, built on the exact axe-v9 base
#  2 entrypoint/cmd/workdir/user and every baked variable equal to axe-v9, except
#    the checkpoint/config paths; the only additions are the TensorRT switches
#  3 size <= 40 GiB
#  4 content: checkpoint/config/manifest hashes = labels, plans = manifest,
#    navsim + nurec_pf + adapter byte-identical to what the 441 run mounted
#    (policy.py differs by the hook only), TensorRT version = engines', no build-only libs
#  5 controller gains through the real CLI validator
#  6 official environment only (4 ALPASIM_* vars, read-only, tmpfs, cap-drop, no
#    network): loads, exact checkpoint, engine active, gRPC get_version answers,
#    VRAM < 16 GiB, no Traceback/FATAL
#  7 the ampere_plus engine (the one non-H100 GPUs would get) deserializes and validates here
#  8 every fallback starts and serves on PyTorch: TensorRT dir missing, switched off,
#    tensorrt not importable, validation rejecting every engine
#  9 in-process test (test_trt_image.py): engine used, same trajectory as PyTorch,
#    per-call and process-wide fallback, latency
# 10 (ECR_URI set) the tag to be pushed is this exact image ID
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
IMAGE="${1:-alpasim-e2e-drivesuprim-stage3:g112-kv28-trt}"
BASE='696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765'
BUNDLE="$ROOT/../models/stage3_vits512_g112_kv28_best_20261010"
SRC="$BUNDLE/AXEv1.0-NuRec"
PREP="$ROOT/runs/prepare-stage3-vits512-g112-kv28-20261010"
GAINS="e2e_challenge/axe_local_eval/controller_gains_axe_v9.json"
SHA=f048fe9c5bc1aaf069d92a023cacb668f2c5e74b2496f107b26e75d347357843
GPU="${GPU:-4}"
OUT="$ROOT/runs/g112-trt-build/validate"
P=g112trt-val
mkdir -p "$OUT"
fail=0
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }
info() { printf '        %s\n' "$*"; }
cleanup() {
    local names; names="$(docker ps -a --filter "name=^${P}-" --format '{{.Names}}')"
    [[ -n "$names" ]] && docker rm -f $names >/dev/null 2>&1
    return 0
}
trap cleanup EXIT
cleanup

echo "image $IMAGE  gpu $GPU  $(date '+%F %T')"

echo "[1] image"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { bad "$IMAGE not found"; exit 1; }
ok "$IMAGE  $(docker image inspect "$IMAGE" --format '{{.Id}}')"
[[ "${IMAGE##*:}" != latest ]] && ok "tag '${IMAGE##*:}'" || bad "'latest' is rejected by the submission CLI"
base_layers="$(docker image inspect "$BASE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')"
img_layers="$(docker image inspect "$IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')"
if [[ "$(head -n "$(wc -l <<<"$base_layers")" <<<"$img_layers")" == "$base_layers" ]]; then
    ok "all $(wc -l <<<"$base_layers") axe-v9 layers, +$(( $(wc -l <<<"$img_layers") - $(wc -l <<<"$base_layers") )) on top"
else
    bad "not built on the axe-v9 base"
fi

echo "[2] runtime config vs axe-v9"
for f in Entrypoint Cmd WorkingDir User; do
    a="$(docker image inspect "$BASE" --format "{{json .Config.$f}}")"
    b="$(docker image inspect "$IMAGE" --format "{{json .Config.$f}}")"
    [[ "$a" == "$b" ]] && ok "$f $b" || bad "$f $b != axe-v9 $a"
done
python3 - "$BASE" "$IMAGE" <<'EOF' || fail=1
import json, subprocess, sys
env = lambda i: dict(e.split("=", 1) for e in json.loads(subprocess.check_output(
    ["docker", "image", "inspect", i, "--format", "{{json .Config.Env}}"])))
base, img = env(sys.argv[1]), env(sys.argv[2])
changed = {"DRIVESUPRIM_CHECKPOINT_PATH": "/app/assets/drivesuprim/pkg_stage3_best.ckpt",
           "DRIVESUPRIM_CONFIG_PATH": "/app/assets/drivesuprim/pkg_config.json"}
added = {"DRIVESUPRIM_TRT_BACKBONE": "auto", "DRIVESUPRIM_TRT_DIR": "/app/assets/drivesuprim/trt"}
problems = []
for k, v in base.items():
    want = changed.get(k, v)
    if img.get(k) != want:
        problems.append(f"{k}={img.get(k)!r} (want {want!r})")
extra = {k: v for k, v in img.items() if k not in base}
if extra != added:
    problems.append(f"added variables {extra} (want {added})")
# The flags the 441 runs depended on, spelled out (the evaluator sets no DRIVESUPRIM_*).
for k, v in {"DRIVESUPRIM_EGO_CENTER_OFFSET": "1", "DRIVESUPRIM_EGO_FOOTPRINT_FROM_API": "1",
             "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION": "1", "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT": "1",
             "DRIVESUPRIM_NAVSIM_2HZ_HISTORY": "1", "DRIVESUPRIM_MAX_BATCH_SIZE": "1",
             "DRIVESUPRIM_DISABLE_INFERENCE": "0", "DRIVESUPRIM_USE_FP16": "0",
             "DRIVESUPRIM_FORCE_PYTORCH_MSDA": "1", "DRIVESUPRIM_PINHOLE_PROFILE": "K377_512X256",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_CACHE_PATH": "/tmp/nv"}.items():
    if img.get(k) != v:
        problems.append(f"{k}={img.get(k)!r} (want {v!r})")
for p in problems:
    print("  FAIL ", p)
if not problems:
    print(f"  PASS  {len(base)} axe-v9 variables kept (2 paths repointed), +{len(added)} TensorRT switches")
sys.exit(1 if problems else 0)
EOF

echo "[3] size"
bytes="$(docker image inspect "$IMAGE" --format '{{.Size}}')"
(( bytes <= 40 * 1024 ** 3 )) && ok "$(numfmt --to=iec "$bytes") <= 40 GiB" || bad "$(numfmt --to=iec "$bytes") > 40 GiB"

echo "[4] content"
labels="$(docker image inspect "$IMAGE" --format '{{json .Config.Labels}}')"
host_sums() {  # relative path + sha256 of every file under the given dirs, no bytecode
    (cd "$1" && shift && find "$@" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum)
}
docker run --rm --name "$P-content" --network none --read-only --entrypoint sh "$IMAGE" -c \
    'cd /app && find navsim nurec_pf drivesuprim_challenge -type f ! -path "*/__pycache__/*" ! -name "*.pyc" -print0 | sort -z | xargs -0 sha256sum' \
    > "$OUT/image_app.sha256"
{ host_sums "$SRC" navsim nurec_pf; host_sums "$PREP" drivesuprim_challenge; } > "$OUT/host_app.sha256"
d="$(diff <(sort -k2 "$OUT/host_app.sha256") <(sort -k2 "$OUT/image_app.sha256") | grep '^[<>]' | awk '{print $1, $3}' | sort -u)"
want=$'< drivesuprim_challenge/policy.py\n> drivesuprim_challenge/policy.py\n> drivesuprim_challenge/trt_backbone_runtime.py'
if [[ "$d" == "$want" ]]; then
    ok "$(wc -l < "$OUT/image_app.sha256") files under navsim, nurec_pf, drivesuprim_challenge identical to the 441 run's mounts except policy.py (+hook) and the new trt_backbone_runtime.py"
else
    bad "code differs from the 441 run's mounts:"; info "$d"
fi
rt="$(grep ' drivesuprim_challenge/trt_backbone_runtime.py$' "$OUT/image_app.sha256" | cut -d' ' -f1)"
[[ "$rt" == "$(sha256sum "$HERE/trt_backbone_runtime.py" | cut -d' ' -f1)" ]] \
    && ok "trt_backbone_runtime.py = $HERE/trt_backbone_runtime.py" \
    || bad "trt_backbone_runtime.py in the image is not the current one (rebuild)"
diff "$PREP/drivesuprim_challenge/policy.py" \
    <(docker run --rm --name "$P-policy" --network none --entrypoint cat "$IMAGE" /app/drivesuprim_challenge/policy.py) \
    > "$OUT/policy.diff"
# the hook: two comment lines, the import, the call (plus blank lines)
if [[ "$(grep -c '^> *[^ ]' "$OUT/policy.diff")" == 4 && "$(grep -c '^<' "$OUT/policy.diff")" == 0 ]] \
   && grep -q '^> *self._trt_engine = _install_trt_backbone(self)$' "$OUT/policy.diff"; then
    ok "policy.py = 441 adapter + the 4-line hook"
else
    bad "policy.py diff unexpected, see $OUT/policy.diff"
fi
docker run -i --rm --name "$P-files" --gpus "device=$GPU" --network none --read-only --entrypoint python \
    -e "LABELS=$labels" -e "EXPECT_CKPT=$SHA" "$IMAGE" - <<'EOF' || fail=1
import hashlib, json, os, sys
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
labels, bad = json.loads(os.environ["LABELS"]), []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        bad.append(name)
ck, cfg = os.environ["DRIVESUPRIM_CHECKPOINT_PATH"], os.environ["DRIVESUPRIM_CONFIG_PATH"]
trt_dir = Path(os.environ["DRIVESUPRIM_TRT_DIR"])
check(f"checkpoint {sha(ck)[:16]}.. = 441 run's and label", sha(ck) == os.environ["EXPECT_CKPT"] == labels["org.alpasim.checkpoint.sha256"])
check("config = label", sha(cfg) == labels["org.alpasim.config.sha256"])
check("vocab present", Path(os.environ["DRIVESUPRIM_VOCAB_PATH"]).is_file())
check("trt manifest = label", sha(trt_dir / "manifest.json") == labels["org.alpasim.trt_manifest.sha256"])
m = json.loads((trt_dir / "manifest.json").read_text())
for e in m["engines"]:
    check(f"engine {e['id']} ({e['bytes'] / 2**20:.0f} MiB) = manifest sha", sha(trt_dir / e["file"]) == e["sha256"])
import tensorrt_bindings as trt
check(f"tensorrt {trt.__version__} = engines' {m['tensorrt_version']}", trt.__version__ == m["tensorrt_version"])
import torch
check(f"torch {torch.__version__} = build's {m['torch_version']}", torch.__version__ == m["torch_version"])
site = Path(trt.__file__).parents[1]
build_only = sorted(p.name for p in (site / "tensorrt_libs").iterdir() if "builder_resource" in p.name)
check(f"no build-only TensorRT libs {build_only}", not build_only)
check("TensorRT Runtime + Logger usable", trt.Runtime(trt.Logger(trt.Logger.ERROR)) is not None)
print("        target", m["target"], "inputs", m["inputs"]["shapes"], "outputs", m["outputs"]["shapes"])
check("checks ran to the end", True)
sys.exit(1 if bad else 0)
EOF

echo "[5] controller gains"
if out="$(uv run python - "$GAINS" <<'PY' 2>&1
import pathlib, sys
sys.path.insert(0, "e2e_challenge/competitor_cli")
from alpasim_challenge import load_controller_gains
g = load_controller_gains(pathlib.Path(sys.argv[1]))
assert (g["lat_position_weight"], g["long_position_weight"], g["idx_start_penalty"]) == (1.0, 0.25, 3), g
assert isinstance(g["idx_start_penalty"], int)
print(" ".join(f"{k}={v}" for k, v in sorted(g.items())))
PY
)"; then ok "$GAINS (lat 1.0 / lon 0.25 / idx 3, as the 441 run)"; info "$out"; else bad "gains: $out"; fi

# Start a driver exactly as the evaluator does (only the four ALPASIM_* variables)
# plus any extra docker args, wait for the policy, ask get_version over gRPC.
OFFICIAL=(--init --gpus "device=$GPU" --network none --read-only --cap-drop ALL
          --security-opt no-new-privileges:true
          --tmpfs /tmp:rw,nosuid,nodev,size=2g --tmpfs /run:rw,nosuid,nodev,size=64m
          -e ALPASIM_DRIVER_HOST=0.0.0.0 -e ALPASIM_DRIVER_PORT=6789
          -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=16)
start() { local name="$P-$1"; shift; docker run -d --name "$name" "${OFFICIAL[@]}" "$@" "$IMAGE" >/dev/null; }
await() {  # name -> 0 when loaded; writes the log and load seconds
    local name="$P-$1" t0=$SECONDS
    while (( SECONDS - t0 < ${START_TIMEOUT:-900} )); do
        docker logs "$name" 2>&1 | grep -q "policy load complete" && { echo $((SECONDS - t0)) > "$OUT/$1.load_s"; break; }
        docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true || break
        sleep 3
    done
    docker logs "$name" > "$OUT/$1.log" 2>&1
    grep -q "policy load complete" "$OUT/$1.log"
}
version() {
    docker exec "$P-$1" python -c '
import grpc
from alpasim_grpc.v0 import common_pb2, egodriver_pb2_grpc
ch = grpc.insecure_channel("localhost:6789")
v = egodriver_pb2_grpc.EgodriverServiceStub(ch).get_version(common_pb2.Empty(), timeout=30)
print(v.version_id, v.grpc_api_version)' 2>&1
}
expect() {  # name label pattern...: every pattern must be in the log, no Traceback/FATAL
    local name=$1 label=$2; shift 2
    local log="$OUT/$name.log" miss=() v=""
    for pat in "$@"; do grep -qE "$pat" "$log" || miss+=("$pat"); done
    grep -qE "Traceback|FATAL" "$log" && miss+=("no Traceback/FATAL")
    if (( ${#miss[@]} == 0 )) && v="$(version "$name")" && [[ "$v" == drivesuprim-e2e-driver* ]]; then
        ok "$label: loaded in $(cat "$OUT/$name.load_s" 2>/dev/null || echo ?)s, get_version '$v'"
        info "$(grep -h 'TRT backbone' "$log" | head -3)"
    else
        bad "$label: missing ${miss[*]:-} ${v:+get_version: $v}"; tail -n 15 "$log" | sed 's/^/        /'
    fi
}

echo "[6] official environment only (GPU $GPU)"
start official
if await official; then
    expect official "official start" "exact checkpoint load: missing=0 unexpected=0" "TRT backbone: active engine=exact"
    pids=" $(docker top "$P-official" -eo pid | tail -n +2 | tr '\n' ' ') "
    mib="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
        while IFS=', ' read -r p m; do [[ "$pids" == *" $p "* ]] && echo "$m"; done | head -1)"
    if [[ -n "$mib" ]] && (( mib < 16384 )); then ok "driver VRAM ${mib} MiB < 16 GiB"; else bad "driver VRAM '${mib:-?}' MiB (limit 16384)"; fi
    sleep 20
    docker inspect -f '{{.State.Running}}' "$P-official" | grep -q true && ok "still serving 20 s later" || bad "exited after load"
else
    bad "official start: no 'policy load complete'"; tail -n 20 "$OUT/official.log" | sed 's/^/        /'
fi
docker rm -f "$P-official" >/dev/null 2>&1

echo "[7]-[8] other engine and fallbacks (GPU $GPU, together)"
SITE=/usr/local/lib/python3.10/dist-packages
start ampere -e DRIVESUPRIM_TRT_ENGINE=ampere_plus
start nodir  -e DRIVESUPRIM_TRT_DIR=/nonexistent
start off    -e DRIVESUPRIM_TRT_BACKBONE=off
start notrt  --tmpfs "$SITE/tensorrt_bindings:rw,size=1m" --tmpfs "$SITE/tensorrt_libs:rw,size=1m"
start reject -e DRIVESUPRIM_TRT_MAX_REL_ERR=0
for n in ampere nodir off notrt reject; do await "$n"; done
expect ampere "ampere_plus engine"        "TRT backbone: active engine=ampere_plus"
expect nodir  "fallback: no engine dir"   "TRT backbone: no manifest"
expect off    "fallback: switched off"    "TRT backbone: disabled"
expect notrt  "fallback: no tensorrt"     "TRT backbone: tensorrt module not importable"
expect reject "fallback: engines rejected" "TRT backbone: every engine rejected"
for n in nodir off notrt reject; do
    grep -q "TRT backbone: active" "$OUT/$n.log" && bad "$n: an engine was activated"
done
cleanup

echo "[9] in-process test (GPU $GPU)"
mkdir -p "$OUT/report" && chmod 777 "$OUT/report"
if docker run --rm --name "$P-test" "${OFFICIAL[@]}" --entrypoint python \
        -v "$HERE/test_trt_image.py:/g112_trt/test_trt_image.py:ro" -v "$OUT/report:/report" \
        "$IMAGE" /g112_trt/test_trt_image.py > "$OUT/test.log" 2>&1; then
    ok "test_trt_image.py passed"
else
    bad "test_trt_image.py failed"; tail -n 25 "$OUT/test.log" | sed 's/^/        /'
fi
[[ -f "$OUT/report/test_trt_image.json" ]] && python3 -c "
import json; r = json.load(open('$OUT/report/test_trt_image.json'))
for k in ('engine','same_selected_candidate','trajectory_max_abs_diff_m','latency_trt','latency_torch',
          'other_shape_answered_by_torch','fallback_on_trt_error','forward_after_fallback_vs_torch_m'):
    print('       ', k, r.get(k))"

echo "[10] tag to push"
if [[ -n "${ECR_URI:-}" ]]; then
    want="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    got="$(docker image inspect "$ECR_URI" --format '{{.Id}}' 2>/dev/null || true)"
    [[ "$got" == "$want" ]] && ok "$ECR_URI is $want" || bad "$ECR_URI is '${got:-untagged}', not $want"
else
    info "ECR_URI not set; skipped (run again with it right before pushing)"
fi

echo
if (( fail )); then echo "VALIDATION FAILED"; exit 1; fi
echo "VALIDATION PASSED"
