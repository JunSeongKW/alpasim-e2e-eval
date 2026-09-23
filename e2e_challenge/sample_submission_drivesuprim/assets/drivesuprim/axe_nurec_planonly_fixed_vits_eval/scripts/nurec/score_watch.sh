#!/usr/bin/env bash
# Score every Nth epoch of a training run as its checkpoints appear.
#
#     source setup_nurec.sh
#     nohup bash scripts/nurec/score_watch.sh > watch.log 2>&1 &
#
# Watches the agent's checkpoint directory, and whenever an epoch it cares about
# lands, runs scripts/nurec/score_epdms.sh over the full val split and appends
# one row to a TSV. It never touches the training process: it only reads files
# the trainer has already written.
#
# Serialised on purpose. Scoring wants a GPU and forty CPU threads, and the
# trainer is using all four cards; two scoring jobs at once would starve it.
# The loop waits for any other run_pdm_score to finish before starting, so it is
# safe to launch while another evaluation is already queued.
#
# Environment:
#   WATCH_EVERY        epoch interval, default 5
#   WATCH_CKPT_DIR     checkpoint directory to watch
#   WATCH_AGENT        agent config name for scoring
#   WATCH_SINCE        only consider checkpoints written after this (date -d string
#                      or @epoch-seconds); default: now, i.e. ignore what already
#                      exists. Set to a run's start time to include its history.
#   WATCH_OUT          TSV to append to
#   WATCH_POLL_S       poll interval, default 120
#   SCORE_THREADS / SCORE_BATCH / SCORE_WORKERS / CUDA_VISIBLE_DEVICES
#                      passed through to score_epdms.sh
set -uo pipefail
cd "${AXE_ROOT:?source setup_nurec.sh first}"

EVERY="${WATCH_EVERY:-5}"
CKPT_DIR="${WATCH_CKPT_DIR:-$NAVSIM_EXP_ROOT/drivesuprim_bevformer_vov_v2_vits_planonly_scratch_nurec_ckpt}"
AGENT="${WATCH_AGENT:-drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec}"
OUT="${WATCH_OUT:-$NUREC_VALIDATION_ROOT/score_watch.tsv}"
POLL="${WATCH_POLL_S:-120}"
SINCE_RAW="${WATCH_SINCE:-now}"
SINCE=$(date -d "$SINCE_RAW" +%s 2>/dev/null) || { echo "bad WATCH_SINCE: $SINCE_RAW" >&2; exit 2; }
STATE="${WATCH_STATE:-$NUREC_VALIDATION_ROOT/.score_watch_done}"

mkdir -p "$(dirname "$OUT")"
touch "$STATE"
[[ -s "$OUT" ]] || printf 'epoch\tstep\tcheckpoint\tframes\tfailed\tscore\tscore_raw\tNC\tDAC\tGT\tEP\tcsv\n' > "$OUT"

echo "watching $CKPT_DIR"
echo "  every $EVERY epochs, agent=$AGENT, since=$(date -d "@$SINCE" '+%F %T')"
echo "  appending to $OUT"

# Scoring is serialised against any other evaluation, including one launched by
# hand.
#
# The match has to be narrowed to actual python processes. `pgrep -f` searches
# whole command lines, and any shell that merely *mentions* the script name --
# an editor invocation, a grep, a wrapper that embeds the same guard -- matches
# too. One such stray shell once blocked a queued job for 46 minutes while no
# scoring was running at all, and it happened to be that job's own parent.
scoring_is_running() {
  local pid
  for pid in $(pgrep -f "run_pdm_score_one_stage_gpu_ssl\.py" 2>/dev/null); do
    [[ "$pid" == "$$" ]] && continue
    case "$(ps -p "$pid" -o comm= 2>/dev/null)" in
      python*) return 0 ;;
    esac
  done
  return 1
}

wait_for_free_gpu() {
  while scoring_is_running; do
    sleep 60
  done
}

summarise() {  # $1 = results csv, $2 = checkpoint
  "$PYTHON_BIN" - "$1" "$2" <<'SUMPY'
import sys, pandas as pd, re, os
csv, ckpt = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
avg = df[df.token == "average_all_frames"]
rows = df[df.token != "average_all_frames"]
if avg.empty:
    raise SystemExit(1)
avg = avg.iloc[0]
name = os.path.basename(ckpt)
m = re.search(r"epoch=(\d+)-step=(\d+)", name)
epoch, step = (m.group(1), m.group(2)) if m else ("?", "?")
def g(col):
    return f"{float(avg[col]):.4f}" if col in avg and pd.notna(avg[col]) else ""
print("\t".join([
    epoch, step, name, str(len(rows)), str(int((~rows["valid"]).sum())),
    g("score"), g("score_raw"), g("no_at_fault_collisions"),
    g("drivable_area_compliance"), g("gt_compliance"), g("ego_progress"), csv,
]))
SUMPY
}

while true; do
  for ckpt in "$CKPT_DIR"/epoch=*.ckpt; do
    [[ -e "$ckpt" ]] || continue
    name=$(basename "$ckpt")
    grep -Fxq "$name" "$STATE" && continue
    # Only this run's checkpoints; the directory is shared across runs and the
    # step numbers repeat, so the write time is what separates them.
    [[ $(stat -c %Y "$ckpt") -lt $SINCE ]] && continue
    epoch=$(sed -n 's/^epoch=0*\([0-9]\+\)-.*/\1/p' <<<"$name")
    [[ -z "$epoch" ]] && epoch=0
    (( (epoch + 1) % EVERY == 0 )) || continue
    # A checkpoint still being written is not worth reading.
    sleep 20
    [[ -s "$ckpt" ]] || continue

    echo "$(date '+%F %T')  scoring epoch $epoch: $name"
    wait_for_free_gpu
    exp="nurec_watch_$(tr '=' '-' <<<"${name%.ckpt}")"
    if SCORE_AGENT="$AGENT" SCORE_SPLIT=val SCORE_EXPERIMENT="$exp" \
       SCORE_THREADS="${SCORE_THREADS:-40}" SCORE_BATCH="${SCORE_BATCH:-8}" \
       SCORE_WORKERS="${SCORE_WORKERS:-8}" MASTER_PORT="${MASTER_PORT:-29800}" \
       bash scripts/nurec/score_epdms.sh "$ckpt" +trainer.params.devices=1 \
       > "/tmp/score_watch_${exp}.log" 2>&1
    then
      csv=$(grep -o "$NAVSIM_EXP_ROOT/$exp/[^ ]*\.csv" "/tmp/score_watch_${exp}.log" | tail -1)
      if [[ -n "$csv" ]] && row=$(summarise "$csv" "$ckpt"); then
        printf '%s\n' "$row" >> "$OUT"
        echo "$(date '+%F %T')  epoch $epoch -> $(cut -f6 <<<"$row") (filter off $(cut -f7 <<<"$row"))"
      else
        echo "$(date '+%F %T')  epoch $epoch: scored but no summary; see /tmp/score_watch_${exp}.log" >&2
      fi
    else
      echo "$(date '+%F %T')  epoch $epoch FAILED; see /tmp/score_watch_${exp}.log" >&2
    fi
    # Recorded either way: a checkpoint that fails to score will fail again, and
    # retrying it forever would block every later epoch behind it.
    echo "$name" >> "$STATE"
  done
  sleep "$POLL"
done
