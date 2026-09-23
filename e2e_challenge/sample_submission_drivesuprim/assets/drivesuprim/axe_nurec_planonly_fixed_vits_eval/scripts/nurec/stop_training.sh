#!/usr/bin/env bash
# Stop a NuRec training run and leave the GPUs clean.
#
# torchrun asks its ranks to exit with SIGTERM. A rank blocked inside a NCCL
# collective never gets to handle it, so torchrun times out and leaves; the
# rank is reparented to init and spins at 100% CPU on a collective whose peers
# are gone, still holding its share of the GPU. nvidia-smi shows 100% util and
# a full card, so the wreck is indistinguishable from a healthy run -- and from
# the terminal it looks like a dataloader that stopped yielding. This kills
# whatever is left, by PID, and reports what the GPUs look like afterwards.
#
#   bash scripts/nurec/stop_training.sh          # ask, then insist
#   DRY_RUN=1 bash scripts/nurec/stop_training.sh
#   TAG=<run_tag> bash scripts/nurec/stop_training.sh   # just that one run
#
# TAG matters once two runs share the machine: without it this clears every
# rank it finds, so a launcher tidying up after itself would take the other
# run down with it. train_staged.sh stamps each launch with ++run_tag=<stem>
# and passes that stem here.
set -uo pipefail

# pgrep -f reads whole command lines, so the pattern matches any shell that
# ever mentioned the script -- this one, the terminal that launched it, an
# editor. Requiring the process to actually be a Python interpreter narrows it
# to the ranks and to torchrun itself, which is what we want gone.
PATTERN='run_training_ssl[.]py'

ranks() {
  local pid comm
  for pid in $(pgrep -f "$PATTERN" 2>/dev/null); do
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    [[ "$comm" == python* || "$comm" == pt_* ]] || continue
    if [[ -n "${TAG:-}" ]]; then
      # The tag rides on the rank's own command line. Workers inherit it, and
      # so does torchrun, so one match covers the whole run and nothing else.
      tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF "$TAG" || continue
    fi
    printf '%s ' "$pid"
  done
}

RANKS=$(ranks)
if [[ -z "${RANKS// /}" ]]; then
  echo "no training ranks running${TAG:+ for tag $TAG}"
else
  echo "ranks: $RANKS"
  ps -o pid=,ppid=,stat=,etime=,pcpu=,rss= -p $RANKS 2>/dev/null
fi

# Dataloader workers die with their rank, but only once the rank is actually
# gone; naming them here means one TERM reaches everyone at the same time.
WORKERS=""
for p in $RANKS; do WORKERS="$WORKERS $(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')"; done
[[ -n "${WORKERS// /}" ]] && echo "workers: $(echo $WORKERS | wc -w)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: nothing signalled"
  exit 0
fi

# Killing every rank on the machine is only ever right when one run owns it.
# Say so explicitly: an untagged call is the one that would take a second run
# down as collateral, and it is exactly the call a stale launcher makes.
if [[ -z "${TAG:-}" && "${ALL:-0}" != "1" ]]; then
  echo "refusing to clear every rank without being asked."
  echo "  this run only:  TAG=<run_tag> bash scripts/nurec/stop_training.sh"
  echo "  everything:     ALL=1 bash scripts/nurec/stop_training.sh"
  exit 3
fi

if [[ -n "${RANKS// /}" ]]; then
  kill -TERM $RANKS $WORKERS 2>/dev/null
  for _ in $(seq 1 "${GRACE_SECONDS:-15}"); do
    [[ -z "$(ranks)" ]] && break
    sleep 1
  done
  LEFT=$(ranks)
  if [[ -n "${LEFT// /}" ]]; then
    echo "still up after SIGTERM: $LEFT -- sending SIGKILL"
    kill -KILL $LEFT $WORKERS 2>/dev/null
    sleep 3
  fi
  LEFT=$(ranks)
  [[ -n "${LEFT// /}" ]] && echo "WARNING: survived SIGKILL: $LEFT" >&2
fi

# Freeing the memory is asynchronous: the driver reclaims a dead process's
# allocation after the kernel tears the context down, which is not instant.
for _ in $(seq 1 20); do
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [[ "$BUSY" -le 4 ]] && break
  sleep 1
done

echo "--- GPUs ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- compute apps ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'
