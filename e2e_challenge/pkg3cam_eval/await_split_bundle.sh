#!/usr/bin/env bash
# Wait for a split bundle (<name>.tar.zst.part-aa, -ab, ...) to finish arriving,
# then extract it over any earlier partial extraction and verify SHA256SUMS.
#
#   await_split_bundle.sh <models_dir> <bundle_name> [timeout_hours]
#
# The parts arrive one at a time and every full part is exactly 950 MiB, so a
# part's size says nothing about whether it is the last. Completeness is decided
# by the stream itself: `zstd -t` over the concatenation succeeds only once the
# final frame is there. It is re-tested whenever a new or grown part has been
# stable for a minute. Prints one line per event; the last line is DONE or FAIL.
set -uo pipefail
DIR="$1"; NAME="$2"; HOURS="${3:-8}"
cd "$DIR" || exit 2
deadline=$(( $(date +%s) + HOURS * 3600 ))
last=""
while (( $(date +%s) < deadline )); do
    state="$(stat -c '%n %s %Y' "$NAME".tar.zst.part-* 2>/dev/null | tr '\n' ' ')"
    newest=$(stat -c %Y "$NAME".tar.zst.part-* 2>/dev/null | sort -n | tail -1)
    if [[ "$state" != "$last" ]] && (( $(date +%s) - newest >= 60 )); then
        last="$state"
        n=$(ls "$NAME".tar.zst.part-* | wc -l)
        if cat "$NAME".tar.zst.part-* | zstd -t -q 2>/dev/null; then
            echo "[$(date '+%F %T')] complete stream with $n parts; extracting"
            cat "$NAME".tar.zst.part-* | zstd -dc | tar -xf - || { echo "FAIL extract"; exit 1; }
            if (cd "$NAME" && sha256sum -c --quiet SHA256SUMS); then
                echo "[$(date '+%F %T')] DONE $NAME: SHA256SUMS OK ($(wc -l < "$NAME/SHA256SUMS") files)"
                exit 0
            fi
            echo "FAIL checksum"; exit 1
        fi
        echo "[$(date '+%F %T')] $n parts, stream still incomplete; waiting"
    fi
    sleep 30
done
echo "FAIL timeout after ${HOURS}h"; exit 1
