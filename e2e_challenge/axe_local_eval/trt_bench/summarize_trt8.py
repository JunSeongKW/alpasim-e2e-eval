"""Summarise the TensorRT speed test: per-call model latency of both arms.

    python summarize_trt8.py --runs ../../../runs --fp32 trt8-fp32-<stamp> --trt trt8-trt-<stamp> \
        [--vram ../../../runs/trt8-vram.csv] [--warmup 3] [--out summary.json]

Each driver prints `[DriveSuprim] TIMING call=N agent_ms=X trt=0|1` per Drive
call (DRIVESUPRIM_TIMING=1): the time of `_run_agent`, synchronised on both
sides, i.e. the model forward the 0.1 s budget is about. The arm script saves
every driver's log next to the run directory before removing the containers.

The first `--warmup` calls of each driver are reported separately (CUDA context,
allocator and kernel selection warm up there) and left out of the steady-state
statistics.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

TIMING = re.compile(r"TIMING call=(\d+) agent_ms=([\d.]+) trt=(\d)")
ARM = "cache-centre-max"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": round(statistics.fmean(xs), 2),
        "p50_ms": round(pct(xs, 0.50), 2),
        "p90_ms": round(pct(xs, 0.90), 2),
        "p99_ms": round(pct(xs, 0.99), 2),
        "max_ms": round(max(xs), 2),
        "min_ms": round(min(xs), 2),
        "std_ms": round(statistics.pstdev(xs), 2),
        "fps_from_mean": round(1000.0 / statistics.fmean(xs), 2),
        "fps_from_p50": round(1000.0 / pct(xs, 0.50), 2),
        "share_over_100ms": round(sum(x > 100.0 for x in xs) / len(xs), 4),
        "headroom_ms_p99_vs_100": round(100.0 - pct(xs, 0.99), 2),
    }


def arm_summary(runs, tag, warmup):
    logs = sorted(glob.glob(os.path.join(runs, f"{tag}-{ARM}.driver-*.log")))
    steady, first, per_driver, trt_flags = [], [], {}, set()
    for path in logs:
        calls = [(int(m.group(1)), float(m.group(2)), m.group(3))
                 for m in map(TIMING.search, open(path, errors="replace")) if m]
        if not calls:
            continue
        calls.sort()
        trt_flags |= {c[2] for c in calls}
        first += [c[1] for c in calls[:warmup]]
        xs = [c[1] for c in calls[warmup:]]
        steady += xs
        if xs:
            per_driver[os.path.basename(path)] = {"n": len(xs), "p50_ms": round(pct(xs, 0.5), 2),
                                                  "mean_ms": round(statistics.fmean(xs), 2)}
    out = {"driver_logs": len(logs), "drivers_with_timing": len(per_driver),
           "trt_flag": sorted(trt_flags)}
    if steady:
        out["steady"] = stats(steady)
        out["warmup_calls_ms"] = [round(x, 1) for x in first]
        p50s = [d["p50_ms"] for d in per_driver.values()]
        out["per_driver_p50_spread_ms"] = [round(min(p50s), 2), round(max(p50s), 2)]
    summ = os.path.join(runs, f"{tag}-{ARM}", "aggregate", "results-summary.json")
    if os.path.exists(summ):
        out["results_summary"] = summ
    prog = os.path.join(runs, f"{tag}.progress.log")
    if os.path.exists(prog):
        walls = re.findall(r"wall_s=(\d+)", open(prog).read())
        if walls:
            out["arm_wall_s"] = int(walls[-1])
    return out


def vram(path):
    peak = defaultdict(int)
    for row in csv.DictReader(open(path)):
        peak[row["container"]] = max(peak[row["container"]], int(row["used_mib"]))
    arms = defaultdict(list)
    for name, mib in peak.items():
        arms["trt" if name.startswith("axe-rr-trt8-trt-") else "fp32"].append(mib)
    return {k: {"drivers": len(v), "peak_mib_mean": round(statistics.fmean(v)),
                "peak_mib_max": max(v)} for k, v in arms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--trt", required=True)
    ap.add_argument("--vram")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = {"fp32": arm_summary(a.runs, a.fp32, a.warmup), "trt": arm_summary(a.runs, a.trt, a.warmup)}
    if "steady" in rep["fp32"] and "steady" in rep["trt"]:
        f, t = rep["fp32"]["steady"], rep["trt"]["steady"]
        rep["speedup"] = {k: round(f[k] / t[k], 3) for k in ("mean_ms", "p50_ms", "p90_ms", "p99_ms")}
        rep["saved_ms"] = {k: round(f[k] - t[k], 2) for k in ("mean_ms", "p50_ms", "p90_ms", "p99_ms")}
    if a.vram and os.path.exists(a.vram):
        rep["vram"] = vram(a.vram)
    s = json.dumps(rep, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
