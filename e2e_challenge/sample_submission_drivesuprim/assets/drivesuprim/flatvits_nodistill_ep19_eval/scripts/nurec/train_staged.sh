#!/usr/bin/env bash
# Train the full ViT-S BEVFormer curriculum on NuRec.
#
# Stage hand-off is weights-only:
#   stage1: perception pre-training, no incoming checkpoint
#   stage2: CKPT=<stage1.ckpt>, planner with the perception trunk frozen
#   stage3: CKPT=<stage2.ckpt>, end-to-end fine-tuning
#
# RESUME_CKPT is different: it resumes an interrupted run of the SAME stage,
# including optimizer, epoch, and scheduler state. Never use it to move between
# stages.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash scripts/nurec/train_staged.sh <stage1|stage2|stage3|planonly> [Hydra overrides...]

Examples:
  NUM_GPUS=4 BATCH_SIZE=8 bash scripts/nurec/train_staged.sh stage1
  CKPT=/path/to/stage1.ckpt NUM_GPUS=4 BATCH_SIZE=8 \
    bash scripts/nurec/train_staged.sh stage2
  CKPT=/path/to/stage2.ckpt NUM_GPUS=4 BATCH_SIZE=8 \
    bash scripts/nurec/train_staged.sh stage3

Resume an interrupted stage (not a stage transition):
  RESUME_CKPT=/path/to/same-stage.ckpt \
    bash scripts/nurec/train_staged.sh stage2

Useful environment variables:
  EPOCHS             override stage default (20 / 5 / 30)
  LR                 default 1e-4
  PRECISION          default bf16-mixed; fp16 is rejected on this pipeline
  NUM_GPUS           default 1
  BATCH_SIZE         per GPU, default 8
  NUM_WORKERS        per rank, default 8
  VAL_BATCHES        validation fraction/count, default 1.0 (the whole val split)
  LOG_EVERY          steps between scalar writes, default 50 (Lightning's own)
  TRAIN_LOG_DIR      where the run transcript goes, default exp/train_logs
  SCORE_VARIANT      nurec (default, NC x DAC x GT x EP) | epdms (old eight heads)
  EXPERIMENT_NAME    Hydra run name
  NUREC_DRY_RUN=1    validate inputs and print the command without launching
  NUREC_SKIP_PREFLIGHT=1  skip the full image/map and token-score checks
EOF
}

# Which score the agent is built for. `nurec` is the default: the labels this
# launcher reads (NUREC_ORI_PDM_SCORE_DIR) are AlpaSim's NC x DAC x GT x EP, and
# the EPDMS agents carry five heads that score would never supervise. Set
# SCORE_VARIANT=epdms to train the old eight-head agents instead -- only
# meaningful against an EPDMS label directory.
SCORE_VARIANT="${SCORE_VARIANT:-nurec}"
case "$SCORE_VARIANT" in
  nurec) AGENT_SUFFIX=_nurec ;;
  epdms) AGENT_SUFFIX="" ;;
  *) echo "Unknown SCORE_VARIANT=$SCORE_VARIANT; use nurec or epdms." >&2; exit 2 ;;
esac

STAGE="${1:-}"
case "$STAGE" in
  stage1)
    AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage1$AGENT_SUFFIX
    DEFAULT_EPOCHS=20
    PREVIOUS_STAGE=""
    ;;
  stage2)
    AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage2$AGENT_SUFFIX
    DEFAULT_EPOCHS=5
    PREVIOUS_STAGE=stage1
    ;;
  stage3)
    AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage3$AGENT_SUFFIX
    DEFAULT_EPOCHS=30
    PREVIOUS_STAGE=stage2
    ;;
  planonly)
    # Not part of the curriculum: the planner alone, from scratch, on the same
    # split and the same labels. It is the baseline the three stages have to
    # beat, so it wants the same epoch budget as stage 3.
    AGENT=drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch$AGENT_SUFFIX
    DEFAULT_EPOCHS=30
    PREVIOUS_STAGE=""
    ;;
  -h|--help|help|"")
    usage
    [[ -n "$STAGE" ]] && exit 0 || exit 2
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
shift

: "${AXE_ROOT:?source setup_nurec.sh first}"
: "${NUREC_PREPARED_ROOT:?source setup_nurec.sh first}"
: "${NUREC_SENSOR_ROOT:?source setup_nurec.sh first}"
: "${NUREC_VOCAB_PATH:?source setup_nurec.sh first}"
: "${NUREC_ORI_PDM_SCORE:?source setup_nurec.sh first}"
: "${NUREC_ORI_PDM_SCORE_DIR:?source setup_nurec.sh first}"
: "${PYTHON_BIN:?source setup_nurec.sh first}"

INDEX="$NUREC_PREPARED_ROOT/index.json"
test -f "$INDEX" || { echo "Missing $INDEX; finish NuRec preparation first" >&2; exit 2; }
test -f "$NUREC_VOCAB_PATH" || { echo "Missing vocabulary: $NUREC_VOCAB_PATH" >&2; exit 2; }
test -d "$NUREC_ORI_PDM_SCORE_DIR" || {
  echo "Missing $NUREC_ORI_PDM_SCORE_DIR; run scripts/nurec/generate_epdms.sh first" >&2
  exit 2
}

if [[ -n "${CKPT:-}" && -n "${RESUME_CKPT:-}" ]]; then
  echo "Set only one of CKPT (previous-stage weights) and RESUME_CKPT (same-stage resume)." >&2
  exit 2
fi
if [[ "$STAGE" == stage1 && -n "${CKPT:-}" ]]; then
  echo "stage1 starts from DINOv3 pre-training and must not receive CKPT." >&2
  exit 2
fi
if [[ -n "$PREVIOUS_STAGE" && -z "${CKPT:-}" && -z "${RESUME_CKPT:-}" ]]; then
  echo "$STAGE requires CKPT=<${PREVIOUS_STAGE}.ckpt> for a weights-only hand-off," >&2
  echo "or RESUME_CKPT=<${STAGE}.ckpt> to continue an interrupted same-stage run." >&2
  exit 2
fi
for checkpoint in "${CKPT:-}" "${RESUME_CKPT:-}"; do
  [[ -z "$checkpoint" || -f "$checkpoint" ]] || {
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
  }
done

# H200 backward is broken with the old torch 2.0.1 environment, and fp16 was
# observed to become NaN around epoch 5-6. Refuse those combinations before DDP
# starts; forward-only smoke tests are not sufficient to expose either issue.
"$PYTHON_BIN" - <<'PY'
import sys
import torch

version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if version < (2, 1):
    raise SystemExit(
        f"Torch {torch.__version__} is unsafe for H200 training. "
        "Use drivesuprim_flash (torch >= 2.1) and re-export PYTHON_BIN."
    )
print(f"torch={torch.__version__} python={sys.executable}")
PY

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

# Clips in the exclusion manifest are not scored -- reverse driving the
# controller cannot follow, broken route/map geometry, and boxes no camera frame
# supports. Their per-token labels therefore do not exist, and the loader opens
# them by name. Filter the split from the same manifest the label job used so
# the two cannot drift apart.
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
print(data.get("map_root", ""))
print(dropped)
SPLITPY
)
echo "excluded clips dropped from the split: ${SPLITS[3]}"

if [[ "${NUREC_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m navsim.planning.data.nurec_validate \
    --log-root "$NUREC_PREPARED_ROOT/navsim_logs/trainval" \
    --sensor-root "$NUREC_SENSOR_ROOT" \
    --map-root "${SPLITS[2]}"

  # Every selected window needs a score. Stage 1 currently multiplies planning
  # loss by zero, but still executes the trajectory graph to give all planner
  # parameters a zero gradient under DDP static_graph=True.
  "$PYTHON_BIN" - "$INDEX" "$NUREC_ORI_PDM_SCORE_DIR" "$NUREC_PREPARED_ROOT" "$STAGE" \
      "$EXCLUSION_MANIFEST" <<'PY'
import json, os, pickle, sys

index, score_dir, prepared, stage, manifest = sys.argv[1:]
logs = json.load(open(index))["logs"]
# The excluded clips carry no labels by design -- the label job never scored
# them -- and the split above already dropped them. Counting their tokens as
# missing labels reports the exclusion manifest back as a failure: 1,440 of
# 47,949, which is exactly the 54 excluded clips.
excluded = set()
if manifest != "none":
    try:
        excluded = set(json.load(open(manifest))["clip_ids"])
    except (FileNotFoundError, ValueError):
        pass
wanted = {log for log in set(logs["train"]) | set(logs["val"])
          if log.removeprefix("nurec-") not in excluded}
scored = {name[:-4] for name in os.listdir(score_dir) if name.endswith(".pkl")}
cache = os.path.realpath(os.path.join(prepared, "metric_cache"))
missing = total = 0
for log in wanted:
    token_dir = os.path.join(cache, log, "unknown")
    if not os.path.isdir(token_dir):
        continue
    for token in os.listdir(token_dir):
        total += 1
        missing += token not in scored
if total == 0:
    raise SystemExit(f"No training tokens found below metric cache {cache}")
if missing:
    raise SystemExit(
        f"{missing} of {total} training tokens have no PDM score in {score_dir}.\n"
        "Wait for label generation or explicitly restrict train_test_split.scene_filter.tokens."
    )

# A map-only stage 1 would silently teach the detector that every pixel is
# empty. Check enough logs to prove that real supported object labels survived
# conversion, without adding a full-dataset pickle scan to every launch.
if stage == "stage1":
    root = os.path.join(prepared, "navsim_logs", "trainval")
    supported = {"vehicle", "pedestrian", "bicycle", "traffic_cone",
                 "barrier", "czone_sign", "generic_object"}
    frames = boxes = 0
    classes = set()
    for log in sorted(wanted)[:100]:
        path = os.path.join(root, f"{log}.pkl")
        with open(path, "rb") as stream:
            for frame in pickle.load(stream):
                names = frame.get("anns", {}).get("gt_names", [])
                frames += 1
                boxes += sum(name in supported for name in names)
                classes.update(name for name in names if name in supported)
        if boxes >= 100:
            break
    if boxes == 0:
        raise SystemExit(
            "Stage 1 preflight found no supported object boxes in the first "
            f"{frames} frames. Refusing to train an all-background detector."
        )
    print(f"stage1 aux targets: sampled_frames={frames} boxes={boxes} classes={sorted(classes)}")
print(f"PDM scores present for all {total} training tokens")
PY
fi

# A NuRec-score agent needs labels carrying gt_compliance; an EPDMS agent needs
# the terms it was built for. Reading one with the other trains heads against a
# column that is not there, which the loss quietly turns into `prediction * 0`.
"$PYTHON_BIN" - "$NUREC_ORI_PDM_SCORE_DIR" "$SCORE_VARIANT" <<'LABELPY' || exit 2
import os, pickle, sys
score_dir, variant = sys.argv[1], sys.argv[2]
names = sorted(os.listdir(score_dir))
if not names:
    raise SystemExit(f"no per-token labels in {score_dir}")
with open(os.path.join(score_dir, names[0]), "rb") as fh:
    keys = set(pickle.load(fh))
if variant == "nurec" and "gt_compliance" not in keys:
    raise SystemExit(
        f"SCORE_VARIANT=nurec needs gt_compliance in the labels; {score_dir} has "
        f"{sorted(keys)}. Point NUREC_ORI_PDM_SCORE_DIR at a NuRec-score label set."
    )
print(f"labels carry: {' '.join(sorted(keys))}")
LABELPY

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
LR="${LR:-1e-4}"
EXPERIMENT="${EXPERIMENT_NAME:-nurec_vits_${SCORE_VARIANT}_${STAGE}}"

# One name for this launch: the transcript is called after it, and it rides on
# every rank's command line as ++run_tag so the teardown can tell this run's
# ranks from a second run sharing the machine.
RUN_TAG="${EXPERIMENT}.${STAGE}.$(date +%Y%m%d-%H%M%S)"

# Two launchers both defaulting to 29500 means the second one dies on bind, so
# take the first free port from there when nothing was asked for.
if [[ -z "${MASTER_PORT:-}" ]]; then
  for port in $(seq 29500 29599); do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$port$"; then
      MASTER_PORT=$port
      break
    fi
  done
  MASTER_PORT="${MASTER_PORT:-29500}"
fi

VOCAB_SIZE=$("$PYTHON_BIN" - "$NUREC_VOCAB_PATH" <<'PY'
import numpy as np, sys
v = np.load(sys.argv[1], mmap_mode="r")
if v.ndim != 3 or v.shape[1:] != (40, 3):
    raise SystemExit(f"Expected vocabulary shape (K,40,3), got {v.shape}")
print(v.shape[0])
PY
)

ARGS=(
  "$AXE_ROOT/navsim/planning/script/run_training_ssl.py"
  "agent=$AGENT"
  "experiment_name=$EXPERIMENT"
  "split=trainval"
  "train_test_split=nurec"
  "navsim_log_path=$NUREC_PREPARED_ROOT/navsim_logs/trainval"
  "original_sensor_path=$NUREC_SENSOR_ROOT"
  "train_logs=${SPLITS[0]}"
  "val_logs=${SPLITS[1]}"
  "dataloader.params.batch_size=$BATCH_SIZE"
  "dataloader.params.num_workers=$NUM_WORKERS"
  "trainer.params.max_epochs=$EPOCHS"
  "trainer.params.precision=$PRECISION"
  # 1.0 = the whole val split, which is what default_training.yaml carries and
  # therefore what the plan-only run validated on. Keep it: val/loss-ori is
  # only comparable across stages if the two runs measure the same clips.
  "trainer.params.limit_val_batches=${VAL_BATCHES:-1.0}"
  "cache_path=${NUREC_CACHE_ROOT:-null}"
  "agent.lr=$LR"
  "++agent.config.vocab_path=$NUREC_VOCAB_PATH"
  "++agent.config.vocab_size=$VOCAB_SIZE"
  "++agent.config.only_ori_input=true"
  "++agent.config.use_traffic_light_compliance=false"
  "++agent.config.ori_vocab_pdm_score_full_path=$NUREC_ORI_PDM_SCORE"
  "++agent.config.ori_vocab_pdm_score_dir=$NUREC_ORI_PDM_SCORE_DIR"
  "++agent.config.use_route=${NUREC_USE_ROUTE:-true}"
)

# Lightning writes no scalar at all until its own log_every_n_steps (50), so at
# a large batch the event file stays empty for minutes and a healthy run is
# indistinguishable from a hung one. Lower it to get an early heartbeat.
# default_training.yaml never names the key and the config is a struct, so a
# plain override is rejected -- ++ adds it. Sent only when asked, so the
# default command line stays exactly the one that has already run.
ARGS+=("++run_tag=$RUN_TAG")

if [[ -n "${LOG_EVERY:-}" ]]; then
  ARGS+=("++trainer.params.log_every_n_steps=$LOG_EVERY")
fi

if [[ -n "${CKPT:-}" ]]; then
  # Keep literal quotes inside the Hydra argument so '=' in Lightning checkpoint
  # filenames is parsed as path text rather than another override.
  ARGS+=("agent.checkpoint_path='$CKPT'")
fi
if [[ -n "${RESUME_CKPT:-}" ]]; then
  ARGS+=("+resume_ckpt_path='$RESUME_CKPT'")
fi
ARGS+=("$@")

echo "stage=$STAGE agent=$AGENT score=$SCORE_VARIANT epochs=$EPOCHS lr=$LR precision=$PRECISION"
echo "gpus=$NUM_GPUS batch_per_gpu=$BATCH_SIZE workers_per_rank=$NUM_WORKERS vocab=$VOCAB_SIZE"
echo "experiment=$EXPERIMENT labels=$NUREC_ORI_PDM_SCORE_DIR"
[[ -n "${CKPT:-}" ]] && echo "init_weights=$CKPT"
[[ -n "${RESUME_CKPT:-}" ]] && echo "resume_state=$RESUME_CKPT"
if [[ "${NUREC_VERBOSE_COMMAND:-0}" == "1" ]]; then
  printf "command:"
  printf " %q" "$PYTHON_BIN" -m torch.distributed.run \
    "--nproc_per_node=$NUM_GPUS" "--master_port=$MASTER_PORT" "${ARGS[@]}"
  printf "\n"
else
  echo "command: $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$NUM_GPUS ... agent=$AGENT"
  echo "         (set NUREC_VERBOSE_COMMAND=1 to print every Hydra override)"
fi

if [[ "${NUREC_DRY_RUN:-0}" == "1" ]]; then
  echo "NUREC_DRY_RUN=1: launch skipped"
  exit 0
fi

# The progress bar and every rank traceback go to stdout/stderr, not to the
# per-run hydra log, so a run watched only through exp/ looks silent whatever
# it is doing, and a rank that dies takes its traceback with it. Keep a
# transcript.
LOG_DIR="${TRAIN_LOG_DIR:-$AXE_ROOT/exp/train_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_TAG}.log"
echo "run_tag: $RUN_TAG   master_port: $MASTER_PORT"
echo "transcript: $LOG_FILE"
echo "  progress: tr '\\r' '\\n' < $LOG_FILE | tail -3"
echo "  stop:     bash $AXE_ROOT/scripts/nurec/stop_training.sh"

set +e
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

# A rank sitting inside a NCCL kernel does not answer the SIGTERM torchrun
# sends on its way out; torchrun gives up and exits anyway. What is left is a
# process reparented to init, holding its GPU and spinning at 100% CPU on a
# collective whose peers are gone -- which reads on nvidia-smi as a run
# training hard, and from the terminal as a dataloader that never yields.
# Ask stop_training.sh rather than pgrep here: pgrep -f matches on whole
# command lines, so the shell that typed this launch counts as a hit and every
# clean run ends on a false alarm. stop_training.sh checks each candidate is a
# Python interpreter before believing it.
# TAG scopes this to the ranks this launcher started. Without it the sweep
# clears every rank on the machine, so a second run sharing the GPUs would be
# killed by whichever launcher happened to exit first.
if DRY_RUN=1 TAG="$RUN_TAG" bash "$AXE_ROOT/scripts/nurec/stop_training.sh" 2>/dev/null \
     | grep -q "^ranks:"; then
  echo "warning: ranks outlived the launcher; clearing them" >&2
  TAG="$RUN_TAG" bash "$AXE_ROOT/scripts/nurec/stop_training.sh" >&2 || true
fi

echo "exit=$STATUS transcript=$LOG_FILE"
exit "$STATUS"
