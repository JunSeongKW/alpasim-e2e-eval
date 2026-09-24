#!/usr/bin/env bash
# One table: what each gamma did to the selection, the metrics and the footage.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
printf '\n%-8s %8s %8s %8s %9s %9s %8s  %s\n' gamma frames changed "cost_max" corridor score dist_m video
for d in runs/gam-g*-cache-centre-max; do
    [[ -d "$d" ]] || continue
    g="$(basename "$d")"; g="${g#gam-g}"; g="${g%-cache-centre-max}"; g="${g//p/.}"
    L=$(cat "$d".driver-*.log 2>/dev/null)
    frames=$(printf '%s' "$L" | grep -c '\] RERANK: ' || true)
    changed=$(printf '%s' "$L" | grep -c '\] RERANK: CHANGED' || true)
    cmax=$(printf '%s' "$L" | grep -o 'sel=#[0-9]* (score=[0-9.]* cost=[0-9.]*' | grep -o 'cost=[0-9.]*$' | cut -d= -f2 | sort -g | tail -1)
    read -r score corr dist < <("$ROOT/.venv/bin/python" - "$d" <<'PY'
import json, sys, pathlib
f = pathlib.Path(sys.argv[1]) / "aggregate" / "results-summary.json"
if not f.exists(): print("- - -"); raise SystemExit
r = [x for x in json.load(open(f))["rollouts"] if x.get("score") is not None]
if not r: print("- - -"); raise SystemExit
m = r[0]["metrics"]
print(f'{r[0]["score"]:.3f}', m.get("left_corridor_laterally"), f'{m.get("dist_traveled_m", 0):.0f}')
PY
)
    v=$(find "$d" -name '*.mp4' 2>/dev/null | grep -v aggregate | head -1)
    printf '%-8s %8s %8s %8s %9s %9s %8s  %s\n' "$g" "$frames" "$changed" "${cmax:--}" "${corr:--}" "${score:--}" "${dist:--}" "${v:-(없음)}"
done
