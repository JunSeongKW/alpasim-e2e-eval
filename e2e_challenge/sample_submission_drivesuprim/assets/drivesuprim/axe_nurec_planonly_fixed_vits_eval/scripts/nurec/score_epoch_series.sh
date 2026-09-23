#!/usr/bin/env bash
# Score a training run's checkpoints every N epochs, waiting for the ones that
# have not been written yet.
#
#     bash scripts/nurec/score_epoch_series.sh 0,5,10,15,20,25,29
#
# Each epoch is copied out of the live checkpoint directory before scoring:
# Lightning writes into a directory named after the agent, so a rerun of the
# same agent -- or a save_top_k that starts pruning -- can remove the file out
# from under an 18-minute job. The copies live under exp/score_ckpts and are
# what the summary refers to.
#
# One line per finished epoch goes to stdout, so this is usable under Monitor:
#
#     epoch=05 score=0.9231 NC=0.9871 DAC=0.9712 GT=0.9950 EP=0.9601
#
# and the same rows accumulate in exp/score_series/summary.tsv.
#
# The val split overlaps train (validation_sampling.overlaps_train), so these
# are not held-out numbers -- they track a run, they do not qualify it.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
AXE=$(pwd)

EPOCHS="${1:-0,5,10,15,20,25,29}"
CKPT_DIR="${CKPT_DIR:-$AXE/exp/drivesuprim_bevformer_vov_v2_vits_planonly_scratch_nurec_ckpt}"
KEEP_DIR="$AXE/exp/score_ckpts"
OUT_DIR="$AXE/exp/score_series"
SUMMARY="$OUT_DIR/summary.tsv"
POLL="${POLL:-120}"
mkdir -p "$KEEP_DIR" "$OUT_DIR"
[ -s "$SUMMARY" ] || printf 'epoch\tscore\tNC\tDAC\tGT\tEP\tcsv\n' > "$SUMMARY"

# shellcheck disable=SC1091
source "$AXE/setup_nurec.sh" > /dev/null 2>&1 || { echo "setup_nurec.sh failed"; exit 2; }

summarise() {  # $1 = csv
  "$PYTHON_BIN" - "$1" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
body = d[d["token"] != "average"] if "token" in d.columns else d
cols = ["score", "no_at_fault_collisions", "drivable_area_compliance",
        "gt_compliance", "ego_progress"]
print("\t".join(f"{body[c].mean():.4f}" for c in cols))
PY
}

IFS=',' read -ra WANT <<< "$EPOCHS"
for ep in "${WANT[@]}"; do
  ep2=$(printf '%02d' "$ep")
  if grep -q "^$ep2	" "$SUMMARY" 2>/dev/null; then
    echo "epoch=$ep2 already scored, skipping"
    continue
  fi

  # Wait for the checkpoint. Training writes it at the end of the epoch, so
  # this is the step that idles for most of the run.
  SRC=""
  while true; do
    SRC=$(ls "$CKPT_DIR"/epoch="$ep2"-step=*.ckpt 2>/dev/null | head -1)
    [ -n "$SRC" ] && break
    if ! pgrep -f "run_training_ssl[.]py" > /dev/null; then
      echo "epoch=$ep2 never arrived and training has stopped; done"
      exit 0
    fi
    sleep "$POLL"
  done

  # Copy before scoring, and only trust the copy once its size matches -- a
  # checkpoint caught mid-write is a truncated file that loads as garbage.
  KEEP="$KEEP_DIR/planonly_fixed_$(basename "$SRC")"
  for _ in $(seq 1 30); do
    cp -f "$SRC" "$KEEP" 2>/dev/null
    [ "$(stat -c %s "$SRC")" = "$(stat -c %s "$KEEP")" ] && break
    sleep 5
  done

  EXP="nurec_score_planonly_ep$ep2"
  SCORE_EXPERIMENT="$EXP" bash "$AXE/scripts/nurec/score_epdms.sh" "$KEEP" \
    > "$OUT_DIR/ep$ep2.log" 2>&1
  CSV=$(find "$AXE/exp/$EXP" -name "*.csv" -newermt "-6 hours" 2>/dev/null | sort | tail -1)
  if [ -z "$CSV" ]; then
    echo "epoch=$ep2 scoring produced no csv -- see $OUT_DIR/ep$ep2.log"
    continue
  fi
  ROW=$(summarise "$CSV")
  printf '%s\t%s\t%s\n' "$ep2" "$ROW" "$CSV" >> "$SUMMARY"
  # shellcheck disable=SC2086
  set -- $ROW
  echo "epoch=$ep2 score=$1 NC=$2 DAC=$3 GT=$4 EP=$5"
done
echo "series complete: $SUMMARY"
