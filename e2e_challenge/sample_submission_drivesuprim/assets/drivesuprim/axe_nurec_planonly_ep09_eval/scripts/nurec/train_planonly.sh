#!/usr/bin/env bash
# Stage-1-free, planning-loss-only training on the rectified NuRec data.
#
#     source setup_nurec.sh
#     NUM_GPUS=4 BATCH_SIZE=8 bash scripts/nurec/train_planonly.sh
#
# The agent defaults to drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch:
# ViT-Small/DINOv3 backbone, planning objective alone, no perception heads
# (use_aux_heads: false), no feasibility gate (feasibility_enabled: false), and
# no stage-1 checkpoint -- everything below the backbone starts from scratch.
# Pass another agent name as $1 to override, extra Hydra overrides after it.
# cache_path defaults to null -- the dataset computes features in the dataloader
# workers instead of reading a pre-built cache.  That is not a size trade
# (though the cache would be ~600 GB: 12.5 MB a sample x 48,038): photometric
# augmentation is applied *inside* compute_features, so a cache freezes one
# jitter/blur draw into every epoch and the student/teacher photometric
# consistency the SSL objective rests on sees the same pair forever.  Set
# NUREC_CACHE_ROOT to re-enable it only if that is what you want.
set -euo pipefail

: "${AXE_ROOT:?source setup_nurec.sh first}"
: "${NUREC_PREPARED_ROOT:?source setup_nurec.sh first}"
: "${NUREC_SENSOR_ROOT:?source setup_nurec.sh first}"
: "${NUREC_VOCAB_PATH:?source setup_nurec.sh first}"
: "${NUREC_ORI_PDM_SCORE_DIR:?source setup_nurec.sh first}"

INDEX="$NUREC_PREPARED_ROOT/index.json"
test -f "$INDEX" || { echo "Missing $INDEX; run the rectify + logs steps first" >&2; exit 2; }
test -d "$NUREC_ORI_PDM_SCORE_DIR" || {
  echo "Missing $NUREC_ORI_PDM_SCORE_DIR; run scripts/nurec/generate_epdms.sh first" >&2; exit 2; }

: "${PYTHON_BIN:?source setup_nurec.sh first}"   # exported by setup_nurec.sh

# Clips in the exclusion manifest are never scored -- reverse driving the
# controller cannot follow, broken route/map geometry, and boxes no camera frame
# supports -- so their per-token labels do not exist and the loader would open
# them by name. Filter the split from the same manifest the label job read.
EXCLUSION_MANIFEST="${NUREC_DAC_EXCLUSION_MANIFEST:-$AXE_ROOT/data/nurec_dac_exclusions.json}"
readarray -t SPLITS < <("$PYTHON_BIN" - "$INDEX" "$EXCLUSION_MANIFEST" <<'SPLITPY'
import json, sys
index, manifest = sys.argv[1], sys.argv[2]
data = json.load(open(index))
excluded = set()
if manifest != "none":
    try:
        excluded = set(json.load(open(manifest))["clip_ids"])
    except FileNotFoundError:
        pass
dropped = 0
for split in ("train", "val"):
    logs = data["logs"][split]
    kept = [log for log in logs if log.removeprefix("nurec-") not in excluded]
    dropped += len(logs) - len(kept)
    print("[" + ",".join(kept) + "]")
print(dropped)
SPLITPY
)
echo "excluded clips dropped from the split: ${SPLITS[2]}"

# Every training window needs its per-token PDM score; a missing one is a hard
# FileNotFoundError deep in the dataloader, so say so here instead.
# The check compares the split against the metric cache, so it has to look at
# the cache the labels were generated against; a different one is an unrelated
# token set and the comparison means nothing. There is one cache now --
# nurec_data/metric_cache, symlinked into the prepared root -- and
# NUREC_METRIC_CACHE_DIRNAME overrides it only for a deliberate comparison.
METRIC_CACHE_DIR="${NUREC_METRIC_CACHE_DIRNAME:-metric_cache}"
"$PYTHON_BIN" - "$INDEX" "$NUREC_ORI_PDM_SCORE_DIR" "$NUREC_PREPARED_ROOT" "$METRIC_CACHE_DIR" "$EXCLUSION_MANIFEST" <<'PY'
import json, os, sys
index, score_dir, prepared, cache_name, manifest = sys.argv[1:6]
scored = {f[:-4] for f in os.listdir(score_dir)}
excluded = set()
if manifest != "none":
    try:
        excluded = set(json.load(open(manifest))["clip_ids"])
    except FileNotFoundError:
        pass
logs = json.load(open(index))["logs"]
wanted = {log for log in set(logs["train"]) | set(logs["val"])
          if log.removeprefix("nurec-") not in excluded}
cache = os.path.realpath(os.path.join(prepared, cache_name))
missing = total = 0
for log in wanted:
    d = os.path.join(cache, log, "unknown")
    if not os.path.isdir(d):
        continue
    for token in os.listdir(d):
        total += 1
        missing += token not in scored
if missing:
    raise SystemExit(
        f"{missing} of {total} training tokens have no PDM score in {score_dir}.\n"
        f"EPDMS is still running -- wait for it, or restrict the run with\n"
        f"  train_test_split.scene_filter.tokens=[...]")
print(f"PDM scores present for all {total} tokens ({cache_name})")
PY

AGENT="${1:-drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXPERIMENT="${EXPERIMENT_NAME:-nurec_k377_${AGENT#drivesuprim_agent_}}"

VOCAB_SIZE=$("$PYTHON_BIN" - "$NUREC_VOCAB_PATH" <<'PY'
import numpy as np, sys
v = np.load(sys.argv[1], mmap_mode="r")
if v.ndim != 3 or v.shape[1:] != (40, 3):
    raise SystemExit(f"Expected vocabulary shape (K,40,3), got {v.shape}")
print(v.shape[0])
PY
)

# RESUME=<checkpoint.ckpt> picks training back up with the optimizer, scheduler,
# epoch and step it was saved with -- not a warm start from the weights. The
# checkpoint is refused if any parameter or optimizer moment in it is non-finite,
# because resuming from NaN weights simply continues producing NaN.
RESUME_OVERRIDE=()
if [[ -n "${RESUME:-}" ]]; then
  test -f "$RESUME" || { echo "no such checkpoint: $RESUME" >&2; exit 2; }
  "$PYTHON_BIN" - "$RESUME" <<'RESUMEPY' || exit 2
import sys, torch
path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
bad = [k for k, v in ckpt["state_dict"].items()
       if v.is_floating_point() and not torch.isfinite(v).all()]
opt = sum(
    1
    for state in ckpt.get("optimizer_states", [])
    for entry in state.get("state", {}).values()
    for v in entry.values()
    if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
)
if bad or opt:
    print(f"Refusing to resume: {len(bad)} non-finite parameters, {opt} non-finite "
          f"optimizer moments in {path}", file=sys.stderr)
    raise SystemExit(2)
print(f"resume from epoch {ckpt.get('epoch')} step {ckpt.get('global_step')} -- weights clean")
RESUMEPY
  RESUME_OVERRIDE=(+resume_ckpt_path="$RESUME")
fi

# default_training.yaml ships precision: 16-mixed, and fp16 produced repeatable
# NaNs on this pipeline -- every loss term, imi included, goes NaN together part
# way through and never recovers, because one inf gradient survives into the
# weights. train_staged.sh has refused fp16 for that reason; this launcher was
# silently inheriting it. Set it here so the two agree.
PRECISION="${PRECISION:-bf16-mixed}"
case "$PRECISION" in
  bf16|bf16-mixed|32|32-true) ;;
  16|16-mixed|fp16|fp16-mixed)
    echo "Refusing PRECISION=$PRECISION: fp16 produced repeatable NaNs on this pipeline." >&2
    exit 2
    ;;
  *)
    echo "Unsupported PRECISION=$PRECISION; use bf16-mixed (recommended) or 32-true." >&2
    exit 2
    ;;
esac

echo "agent=$AGENT  gpus=$NUM_GPUS  batch=$BATCH_SIZE  vocab=$VOCAB_SIZE  precision=$PRECISION  resume=${RESUME:-none}  exp=$EXPERIMENT"

# torch.distributed.run rather than the torchrun shim: the shim lives in the
# conda env's bin and is only on PATH once that env is activated, while
# PYTHON_BIN is already resolved above.
ARGS=(
  "$AXE_ROOT/navsim/planning/script/run_training_ssl.py"
  agent="$AGENT" experiment_name="$EXPERIMENT"
  split=trainval train_test_split=nurec
  navsim_log_path="$NUREC_PREPARED_ROOT/navsim_logs/trainval"
  original_sensor_path="$NUREC_SENSOR_ROOT"
  "train_logs=${SPLITS[0]}" "val_logs=${SPLITS[1]}"
  dataloader.params.batch_size="$BATCH_SIZE"
  trainer.params.precision="$PRECISION"
  "${RESUME_OVERRIDE[@]}"
  cache_path="${NUREC_CACHE_ROOT:-null}"
  dataloader.params.num_workers="${NUM_WORKERS:-8}"
  ++agent.config.vocab_path="$NUREC_VOCAB_PATH"
  ++agent.config.vocab_size="$VOCAB_SIZE"
  ++agent.config.only_ori_input=true
  ++agent.config.use_traffic_light_compliance=false
  ++agent.config.ori_vocab_pdm_score_full_path="$NUREC_ORI_PDM_SCORE"
  ++agent.config.ori_vocab_pdm_score_dir="$NUREC_ORI_PDM_SCORE_DIR"
  ++agent.config.use_route="${NUREC_USE_ROUTE:-true}"
  "${@:2}"
)

if [[ "${NUREC_DRY_RUN:-0}" == "1" ]]; then
  printf '  %s\n' "${ARGS[@]}"
  echo "NUREC_DRY_RUN=1: launch skipped"
  exit 0
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" \
  "${ARGS[@]}"
