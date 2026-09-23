#!/usr/bin/env bash
# Environment for the NuRec-only AXE tree.  Source it, do not execute it:
#
#     source setup_nurec.sh
#
# Every path resolves from this directory.  Data that is large, still being
# written, or owned by someone else is reached through symlinks under ./data
# rather than copied, so this tree is the single place to look without being
# the single place that stores 100 GB.
#
#   ./data/prepared         the K377 prepared root the model trains from
#   ./data/prepared_source  the un-rectified root (maps / routes / metric_cache)
#   ./data/images           the 512x256 rectified PNGs (K377, from the
#                           challenge_jpeg95_1920x1080 render)
#   ./data/epdms            per-token PDM scores (written by generate_epdms.sh)
#   ./data/validation       verification artefacts and figures
#   ./data/renders_raw      the challenge-matched raw render (read-only)
#   ./data/usdz             the NuRec sample set (read-only)
#   ./exp                   experiment outputs
#   ./assets/nurec          vocabulary and calibration this tree owns

_this="${BASH_SOURCE[0]:-$0}"
export AXE_ROOT="$(cd "$(dirname "$_this")" && pwd)"
export WORKSPACE_ROOT="$(cd "$AXE_ROOT/.." && pwd)"

# ---------------------------------------------------------------- devkit ----
# NAVSIM_DEVKIT_ROOT must be this tree.  It used to point at ../DriveSuprim,
# which resolved a different copy of the same package -- so the code that ran
# was not the code being edited.
export NAVSIM_DEVKIT_ROOT="$AXE_ROOT"
export PYTHONPATH="$AXE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$AXE_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$AXE_ROOT/data}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$WORKSPACE_ROOT/dataset/maps}"

# ------------------------------------------------------------------ NuRec ----
# Every path below is ${VAR:-default}, so a value already exported in the shell
# wins over the default. That is deliberate -- it is how a run points at an
# alternative -- but it also means a stale export from an earlier session
# silently overrides a corrected default, and the failure surfaces far away as
# "finish NuRec preparation first". Say what actually happened instead.
_stale=""
for _v in NUREC_PREPARED_ROOT NUREC_SOURCE_ROOT NUREC_ORI_PDM_SCORE_DIR \
          NUREC_METRIC_CACHE_ROOT NUREC_VOCAB_PATH; do
  _val="${!_v-}"
  if [[ -n "$_val" && ! -e "$_val" ]]; then
    _stale+="  $_v=$_val"$'\n'
  fi
done
if [[ -n "$_stale" ]]; then
  echo "These are exported in this shell and do not exist:" >&2
  printf '%s' "$_stale" >&2
  echo "They override the defaults in setup_nurec.sh. Clear them and re-source:" >&2
  echo "  unset NUREC_PREPARED_ROOT NUREC_SOURCE_ROOT NUREC_ORI_PDM_SCORE \\" >&2
  echo "        NUREC_ORI_PDM_SCORE_DIR NUREC_EPDMS_NAME NUREC_METRIC_CACHE_ROOT \\" >&2
  echo "        NUREC_METRIC_CACHE_DIRNAME NUREC_VOCAB_PATH" >&2
fi
unset _stale _v _val

export NUREC_PREPARED_ROOT="${NUREC_PREPARED_ROOT:-$AXE_ROOT/data/prepared}"
export NUREC_SOURCE_ROOT="${NUREC_SOURCE_ROOT:-$AXE_ROOT/data/prepared_source}"
export NUREC_SENSOR_ROOT="${NUREC_SENSOR_ROOT:-$AXE_ROOT/data/images}"
export NUREC_MAP_ROOT="${NUREC_MAP_ROOT:-$AXE_ROOT/data/prepared_source/maps}"
export NUREC_CALIBRATION="${NUREC_CALIBRATION:-$AXE_ROOT/assets/nurec/calibration.json}"
export NUREC_VALIDATION_ROOT="${NUREC_VALIDATION_ROOT:-$AXE_ROOT/data/validation}"

# Source data, read-only.
export NUREC_USDZ_ROOT="${NUREC_USDZ_ROOT:-$AXE_ROOT/data/usdz}"
export NUREC_RENDER_ROOT="${NUREC_RENDER_ROOT:-$AXE_ROOT/data/renders_raw}"

# Trajectory vocabulary and the per-token PDM scores the planning loss reads.
#
# Two 4096-entry vocabularies are in play and they are NOT interchangeable:
#
#   nurec_train_kmeans_4096x40x3.npy   fitted on NuRec, reaches 159.2 m at 4 s
#   test_4096_kmeans.npy               NAVSIM's, reaches 58.7 m at 4 s
# (4 s is pose index 39: 40 poses at 0.1 s.  The 31.6 / 11.7 figures that used to
#  sit here are index 7, which is 0.8 s -- the GT trajectory's 0.5 s spacing read
#  onto a vocabulary that does not use it.  See docs/nurec/PAIRING.md.)
#
# The per-token PDM scores are an array indexed by vocabulary entry, so a score
# file and the vocabulary it was generated against are one artefact. Pairing a
# model with the wrong one silently trains it towards different trajectories
# than the scores describe.
#
#   NUREC_VOCAB=nurec    (default) the NuRec vocabulary -- what the EPDMS run
#                        generating data/epdms is scoring, and what training
#                        from 2026-08-26 onward uses
#   NUREC_VOCAB=navsim   NAVSIM's -- required to evaluate checkpoints trained
#                        before that, including epoch=03-step=2656
export NUREC_VOCAB="${NUREC_VOCAB:-nurec}"
# The controller follows the vocabulary, because they mark the same two eras and
# splitting them is how you end up scoring a model against targets built under a
# different controller.  NAVSIM-era work is LQR over a kinematic bicycle
# throughout; NuRec-era work is the Runtime's linear MPC over its dynamic
# bicycle.  NUREC_USE_MPC still overrides if you really mean to cross them.
case "$NUREC_VOCAB" in
  nurec)  _vocab_file="nurec_train_kmeans_4096x40x3.npy"; _mpc=1 ;;
  navsim) _vocab_file="test_4096_kmeans.npy";             _mpc=0 ;;
  *) echo "  NUREC_VOCAB must be 'nurec' or 'navsim', got '$NUREC_VOCAB'"; _mpc=1 ;;
esac
export NUREC_USE_MPC="${NUREC_USE_MPC:-$_mpc}"
unset _mpc
export NUREC_VOCAB_PATH="${NUREC_VOCAB_PATH:-$AXE_ROOT/assets/nurec/vocab/$_vocab_file}"
unset _vocab_file
export NAVSIM_TRAJPDM_ROOT="${NAVSIM_TRAJPDM_ROOT:-$AXE_ROOT/data/epdms}"
# The label set the NuRec score is trained and evaluated against: gt_compliance
# and the AlpaSim DAC rule. There is exactly one, and it is the default -- the
# superseded sets live under _archive_*/ rather than beside it with a version
# suffix, because a stale default that every run overrode by hand is how the
# wrong labels stay in a config unnoticed.
export NUREC_EPDMS_NAME="${NUREC_EPDMS_NAME:-nurec}"
_score_dir="$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_4096_$NUREC_EPDMS_NAME"
export NUREC_ORI_PDM_SCORE="${NUREC_ORI_PDM_SCORE:-$_score_dir/${NUREC_EPDMS_NAME}_epdms.pkl}"
export NUREC_ORI_PDM_SCORE_DIR="${NUREC_ORI_PDM_SCORE_DIR:-$_score_dir/per_token}"
unset _score_dir

# ----------------------------------------------------------------- route ----
# The PAI-Track route replaces the 4-way driving_command as the intent signal.
# NuRec ships a real route in every prepared log (route_waypoints; 99.7% of
# frames carry one), so it is ON by default here -- this is a NuRec-only
# default, the DriveSuprimConfig dataclass field stays False so the six NAVSIM
# evaluations keep using the command.
#
# It is set HERE, once, rather than per script, because training and scoring
# must agree.  With use_route=True the first four columns of _status_encoding
# get an exactly-zero gradient and stay at their init; scoring the resulting
# checkpoint with use_route=False multiplies a real driving_command into those
# never-trained weights.  Nothing raises -- the number just gets worse.  Both
# scripts/nurec/train_planonly.sh and scripts/nurec/score_epdms.sh read this
# variable, so flipping it flips both together.
export NUREC_USE_ROUTE="${NUREC_USE_ROUTE:-true}"

# ------------------------------------------------------- challenge driver ----
# The submission the geometry tests compare against.  Two trees carry a copy;
# this one has the K377 patch (2026-08-25), the other is still at 412/366 and
# would make test_the_target_pinhole_is_the_drivers fail.
export NUREC_CHALLENGE_DRIVER="${NUREC_CHALLENGE_DRIVER:-/rhome/junhyeok/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/sample_submission_drivesuprim/drivesuprim_challenge/driver.py}"
export NUREC_CHALLENGE_RECTIFICATION="${NUREC_CHALLENGE_RECTIFICATION:-/rhome/junhyeok/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/sample_submission_drivesuprim/drivesuprim_challenge/rectification.py}"

# ------------------------------------------------------------------- NCCL ----
# Without a usable interface NCCL fails to bootstrap on this host
# ("Bootstrap : no socket interface found"), even for a single rank.  The
# ambient value is not trusted: this login environment ships
# NCCL_SOCKET_IFNAME=eno1, and eno1 is DOWN with no address -- which is exactly
# the bootstrap failure.  An inherited name is kept only if it is up and
# addressed, otherwise replaced by the interface the default route uses.
_nccl_usable() {
  [ -n "$1" ] && ip -brief addr show "$1" 2>/dev/null | grep -q "UP.*[0-9]\{1,3\}\."
}
if ! _nccl_usable "${NCCL_SOCKET_IFNAME:-}"; then
  _iface="$(ip route get 8.8.8.8 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [ -n "${NCCL_SOCKET_IFNAME:-}" ] && \
    echo "  note: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} is not up; using ${_iface:-lo}"
  export NCCL_SOCKET_IFNAME="${_iface:-lo}"
  unset _iface
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ---------------------------------------------------------------- python ----
# `command -v python` finds /usr/bin/python3.8 on this host, which has none of
# the dependencies.  Pin the interpreter the pipeline was verified against.
if [ -z "${PYTHON_BIN:-}" ]; then
  for _c in "$HOME/miniconda3/envs/drivesuprim_flash/bin/python" \
            "$HOME/.conda/envs/drivesuprim_flash/bin/python" \
            "$HOME/.conda/envs/drivesuprim/bin/python" \
            "/rhome/satyam/.conda/envs/drivesuprim/bin/python"; do
    [ -x "$_c" ] && { export PYTHON_BIN="$_c"; break; }
  done
  export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
  unset _c
fi
unset -f _nccl_usable

# ------------------------------------------------------------------ report ---
echo "AXE (NuRec)  $AXE_ROOT   vocab=$NUREC_VOCAB   controller=$([ "$NUREC_USE_MPC" = 1 ] && echo MPC || echo LQR)   route=$([ "$NUREC_USE_ROUTE" = true ] && echo on || echo off)"
_missing=0
_check() {  # label path [optional]
  if [ -e "$2" ]; then
    printf "  %-22s %s\n" "$1" "ok"
  elif [ "${3:-}" = "optional" ]; then
    printf "  %-22s %s\n" "$1" "not yet (${2##*/})"
  else
    printf "  %-22s %s\n" "$1" "MISSING $2"; _missing=$((_missing+1))
  fi
}
_check "prepared root"   "$NUREC_PREPARED_ROOT/index.json" optional
_check "rectified images" "$NUREC_SENSOR_ROOT"
_check "maps"            "$NUREC_MAP_ROOT"
_check "calibration"     "$NUREC_CALIBRATION"
_check "vocabulary"      "$NUREC_VOCAB_PATH"
_check "pdm per-token"   "$NUREC_ORI_PDM_SCORE_DIR"
_check "pdm aggregate"   "$NUREC_ORI_PDM_SCORE" optional
_check "raw renders"     "$NUREC_RENDER_ROOT"
_check "usdz"            "$NUREC_USDZ_ROOT"
_check "nccl interface"  "/sys/class/net/$NCCL_SOCKET_IFNAME"
[ "$_missing" -gt 0 ] && echo "  -> $_missing missing; see docs/nurec/SETUP.md"
unset _missing _this; unset -f _check
