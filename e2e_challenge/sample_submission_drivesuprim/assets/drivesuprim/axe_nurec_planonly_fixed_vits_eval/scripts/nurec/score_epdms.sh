#!/usr/bin/env bash
# Score a trained checkpoint's own trajectories with EPDMS on the NuRec split.
#
#     source setup_nurec.sh
#     bash scripts/nurec/score_epdms.sh /path/to/epoch=NN-step=NNNN.ckpt
#
# This is not the same job as scripts/nurec/generate_epdms.sh. That one scores
# all 4,096 vocabulary trajectories per token to build the training target; this
# one runs the model, takes the trajectory it picks, and scores that -- the
# number you would quote for the checkpoint.
#
# The column to read is `score`, which on NuRec scenes is the AlpaSim scene
# score NC x DAC x GT x EP -- the same aggregation the training labels use. The
# CSV also carries `pdms` and `pdms_v1`, the old weighted EPDMS sums, kept for
# comparison against previously published numbers; they are NOT what this model
# was trained against and read far higher, because the terms that no longer
# enter the score sit pinned at 1.0 inside their weighted average.
#
# CAVEAT, and it is not a small one: the val split in index.json is drawn from
# the train logs (`validation_sampling.overlaps_train: true`), so by default this
# scores the model on scenes it trained on. Useful for watching a run move; not a
# generalisation figure. Pass SCORE_SPLIT=train to be explicit that it is the
# same data, or point --log-names at a held-out set once one exists.
set -euo pipefail

: "${AXE_ROOT:?source setup_nurec.sh first}"
: "${PYTHON_BIN:?source setup_nurec.sh first}"

CKPT="${1:?usage: score_epdms.sh <checkpoint.ckpt> [extra hydra overrides]}"
test -f "$CKPT" || { echo "no such checkpoint: $CKPT" >&2; exit 2; }

AGENT="${SCORE_AGENT:-drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec}"

# The controller has to be the one the training targets were scored with, or the
# numbers are not comparable.  setup_nurec.sh ties NUREC_USE_MPC to NUREC_VOCAB
# (navsim -> LQR, nurec -> MPC) so one switch sets both; this only reads it.
SIMULATOR_OVERRIDE=()
if [[ "${NUREC_USE_MPC:-1}" == "1" ]]; then
  SIMULATOR_OVERRIDE=(
    simulator._target_=navsim.planning.simulation.planner.nurec_controller.nurec_simulator.NuRecSimulator
  )
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUREC_CONTROLLER_DTYPE="${NUREC_CONTROLLER_DTYPE:-float64}"
fi
SPLIT="${SCORE_SPLIT:-val}"            # val | train
# SCORE_UNUSED=1 also measures the terms the score dropped (TTC, lane
# keeping, comfort, driving direction), which is what `pdms` / `pdms_v1`
# weigh. Roughly 5x slower. Left off, those columns come out blank rather
# than as a weighted average over values held at 1.0.
UNUSED_OVERRIDE=()
if [[ "${SCORE_UNUSED:-0}" == "1" ]]; then
  UNUSED_OVERRIDE=(++scorer.config.score_unused_metrics=true)
fi
# SCORE_SAVE_PICKLE=1 keeps the trajectory the model picked for every token, as
# <ckpt-dir>/<ckpt-name>.pkl. scripts/nurec/score_video.py needs it: the video
# draws what was actually graded rather than re-running the model, so the frames
# and the CSV cannot disagree. Off by default -- it is a large file per
# checkpoint and nothing but the video reads it.
TRAFFIC="${TRAFFIC_AGENTS:-non_reactive}"
# Scoring is two phases with different bottlenecks.
#
#   1. prediction  -- the model over every scene, on the GPUs.  The shipped
#      config runs batch_size 1, which leaves the cards at 15-40% while the
#      dataloader feeds them one sample at a time.
#   2. simulation  -- rolling each trajectory out through the controller and
#      scoring it.  Pure CPU, distributed by Ray over threads_per_node.
#
# Defaults here are sized for a 36-core box with four free 24 GB cards. Drop
# them if you are sharing.
THREADS="${SCORE_THREADS:-24}"
SCORE_BATCH="${SCORE_BATCH:-8}"
SCORE_WORKERS="${SCORE_WORKERS:-8}"
INDEX="$NUREC_PREPARED_ROOT/index.json"
# Must be the cache the labels were scored against, or the observations, road
# edges and recorded run differ between target and evaluation.  Same variable
# the training launcher reads, so one export covers both.
# The AlpaSim-rule cache. It lives at nurec_data/metric_cache and is symlinked
# into the prepared root, because moving NUREC_PREPARED_ROOT to reach a cache is
# what broke a 30-epoch run: the root it was moved to carried the
# pre-rectification nuPlan pinhole against rectified 512x256 files, so every BEV
# reference point projected outside the image (bev_mask 0.000%) and the encoder
# trained and scored on no image content at all. Point the CACHE somewhere else,
# never the root. DriveSuprimFeatureBuilder now refuses the mismatched pair.
METRIC_CACHE_DIRNAME="${NUREC_METRIC_CACHE_DIRNAME:-metric_cache}"
METRIC_ROOT="${NUREC_METRIC_CACHE_ROOT:-$NUREC_PREPARED_ROOT/$METRIC_CACHE_DIRNAME}"
test -f "$INDEX" || { echo "Missing $INDEX; run finalize_data.sh first" >&2; exit 2; }
test -d "$METRIC_ROOT" || { echo "Missing $METRIC_ROOT" >&2; exit 2; }

# Clips in the exclusion manifest were never scored, so they carry no labels
# and were dropped from training. Scoring them here would mix in frames the
# model was never shown, against a recorded run the controller cannot follow.
EXCLUSION_MANIFEST="${NUREC_DAC_EXCLUSION_MANIFEST:-$AXE_ROOT/data/nurec_dac_exclusions.json}"

readarray -t INFO < <("$PYTHON_BIN" - "$INDEX" "$SPLIT" "$EXCLUSION_MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
logs = d["logs"][sys.argv[2]]
manifest = sys.argv[3]
excluded = set()
if manifest != "none":
    try:
        excluded = set(json.load(open(manifest))["clip_ids"])
    except FileNotFoundError:
        pass
kept = [log for log in logs if log.removeprefix("nurec-") not in excluded]
print("[" + ",".join(kept) + "]")
print(len(kept))
overlap = d.get("validation_sampling", {}).get("overlaps_train")
print("yes" if overlap else "no")
print(len(logs) - len(kept))
PY
)
echo "agent=$AGENT  split=$SPLIT  logs=${INFO[1]} (excluded ${INFO[3]})  traffic=$TRAFFIC  "\
     "cache=$(basename "$METRIC_ROOT")  unused_metrics=$([[ "${SCORE_UNUSED:-0}" == 1 ]] && echo on || echo off)  controller=$([[ "${NUREC_USE_MPC:-1}" == 1 ]] && echo MPC || echo LQR)  "\
     "batch=$SCORE_BATCH  workers=$SCORE_WORKERS  ray_threads=$THREADS"
[ "${INFO[2]}" = "yes" ] && [ "$SPLIT" = "val" ] && \
  echo "  NOTE: this val split overlaps train -- the score is not held out."

VOCAB_SIZE=$("$PYTHON_BIN" - "$NUREC_VOCAB_PATH" <<'PY'
import numpy as np, sys
v = np.load(sys.argv[1], mmap_mode="r")
print(v.shape[0])
PY
)

# Checkpoint names carry "=" (epoch=01-step=0218). Hydra's override grammar
# rejects a bare "=" inside a value, so the path is passed single-quoted for
# Hydra (the shell quotes are consumed before it ever sees them) and the "=" is
# stripped from the experiment name.
CKPT_TAG="$(basename "$CKPT" .ckpt | tr '=' '-')"
EXPERIMENT="${SCORE_EXPERIMENT:-nurec_epdms_score_${CKPT_TAG}}"

"$PYTHON_BIN" "$AXE_ROOT/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py" \
  "${SIMULATOR_OVERRIDE[@]}" \
  "${UNUSED_OVERRIDE[@]}" \
  agent="$AGENT" \
  agent.checkpoint_path="'$CKPT'" \
  experiment_name="$EXPERIMENT" \
  train_test_split=nurec \
  navsim_log_path="$NUREC_PREPARED_ROOT/navsim_logs/trainval" \
  original_sensor_path="$NUREC_SENSOR_ROOT" \
  metric_cache_path="$METRIC_ROOT" \
  "train_test_split.scene_filter.log_names=${INFO[0]}" \
  traffic_agents="$TRAFFIC" \
  ++agent.config.vocab_path="$NUREC_VOCAB_PATH" \
  ++agent.config.vocab_size="$VOCAB_SIZE" \
  ++agent.config.only_ori_input=true \
  ++agent.config.use_traffic_light_compliance=false \
  ++agent.config.use_route="${NUREC_USE_ROUTE:-true}" \
  ++agent.config.training=false \
  ++agent.config.inference.save_pickle="${SCORE_SAVE_PICKLE:-false}" \
  dataloader.params.batch_size="$SCORE_BATCH" \
  dataloader.params.num_workers="$SCORE_WORKERS" \
  worker.threads_per_node="$THREADS" \
  "${@:2}"
