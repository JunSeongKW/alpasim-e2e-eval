#!/usr/bin/env bash
# Re-render the forty validation clips for a different checkpoint.
#
#     source setup_nurec.sh
#     bash scripts/nurec/render_val40.sh <checkpoint.ckpt> <out-name>
#
# The clip list is taken from an existing render (epoch29_val40 by default) so
# the two sets show the same forty scenes and can be watched side by side. Pass
# CLIP_LIST=<file of clip ids, one per line> to use a different set.
#
# score_video.py draws the trajectory out of the scoring job's prediction
# pickle, not by re-running the model, so the pickle has to exist. If it does
# not, this scores the checkpoint once with SCORE_SAVE_PICKLE=1 to produce it
# along with the CSV the score overlay reads.
set -uo pipefail
cd "${AXE_ROOT:?source setup_nurec.sh first}"

CKPT="${1:?usage: render_val40.sh <checkpoint.ckpt> <out-name>}"
NAME="${2:?usage: render_val40.sh <checkpoint.ckpt> <out-name>}"
test -f "$CKPT" || { echo "no such checkpoint: $CKPT" >&2; exit 2; }

REF_DIR="${REF_DIR:-$AXE_ROOT/data/validation/score_video/epoch29_val40}"
CLIP_LIST="${CLIP_LIST:-}"
OUT_DIR="$AXE_ROOT/data/validation/score_video/$NAME"
PKL="$(dirname "$CKPT")/$(basename "$CKPT" .ckpt).pkl"
EXP="${SCORE_EXPERIMENT:-nurec_video_${NAME}}"
mkdir -p "$OUT_DIR"

if [ -n "$CLIP_LIST" ]; then
  readarray -t CLIPS < "$CLIP_LIST"
else
  readarray -t CLIPS < <(find "$REF_DIR" -name "*.mp4" -printf "%f\n" | sed 's/\.mp4$//' | sort)
fi
[ "${#CLIPS[@]}" -gt 0 ] || { echo "no clips found (REF_DIR=$REF_DIR)" >&2; exit 2; }
echo "clips: ${#CLIPS[@]}   out: $OUT_DIR"

# The scoring CSV for this checkpoint. Reuse one if a previous job left it and
# the pickle beside it, otherwise score once with the pickle turned on.
CSV=$(find "$AXE_ROOT/exp/$EXP" -name "*.csv" 2>/dev/null | sort | tail -1)
if [ ! -f "$PKL" ] || [ -z "$CSV" ]; then
  echo "scoring $(basename "$CKPT") to produce the prediction pickle..."
  SCORE_SAVE_PICKLE=1 SCORE_EXPERIMENT="$EXP" \
    bash "$AXE_ROOT/scripts/nurec/score_epdms.sh" "$CKPT" \
    > "$AXE_ROOT/exp/${EXP}_score.log" 2>&1
  CSV=$(find "$AXE_ROOT/exp/$EXP" -name "*.csv" 2>/dev/null | sort | tail -1)
fi
test -f "$PKL" || { echo "no prediction pickle at $PKL" >&2; exit 3; }
test -n "$CSV"  || { echo "no scoring csv under exp/$EXP" >&2; exit 3; }
echo "predictions: $PKL"
echo "scores:      $CSV"

OK=0; FAIL=0
for clip in "${CLIPS[@]}"; do
  [ -n "$clip" ] || continue
  OUT="$OUT_DIR/$clip.mp4"
  [ -s "$OUT" ] && { OK=$((OK+1)); continue; }
  if "$PYTHON_BIN" "$AXE_ROOT/scripts/nurec/score_video.py" \
        --predictions "$PKL" --scores "$CSV" \
        --log "nurec-$clip" --out "$OUT" > "$OUT_DIR/$clip.log" 2>&1; then
    OK=$((OK+1))
    echo "rendered $clip  ($OK/${#CLIPS[@]})"
  else
    FAIL=$((FAIL+1))
    echo "FAILED $clip -- $(tail -2 "$OUT_DIR/$clip.log" | tr '\n' ' ')"
  fi
done
echo "done: $OK rendered, $FAIL failed, in $OUT_DIR"
