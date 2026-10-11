"""Speed through each stretch of a curve, policy vs the logged human.

    .venv/bin/python e2e_challenge/axe_local_eval/curve_speed_profile.py \
        --run g112_kv28=runs/leaderboard-stage3-vits512-g112-kv28-20261010 \
        --run axe-v9=runs/curve-axe-v9-441-20261010 --out runs/curve-stop-vs-human-20261011

Positions are relative to the start of the curve (the 15 m window with the
largest heading change, as in curve_stop_vs_human.py), from 30 m before it to
30 m after. The speed through a 2 m bin is 2 m over the time spent crossing it,
so a stop inside the bin shows as a low speed rather than vanishing from an
average over samples. A policy bin counts only if it lies wholly inside the
policy's own driving (after the forced-GT start, before any collision); a bin the
policy entered but never left before the clip ended is kept with its censored
speed, which is what a stall looks like.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curve_stop_vs_human import CURVE_DEG, FEATURES, USDZ, human_log, load_run, usdz_paths  # noqa: E402

BIN_M = 2.0
EDGES = np.arange(-30.0, 30.0 + BIN_M, BIN_M)
SEGMENTS = {"approach (-20..0 m)": (-20.0, 0.0), "in curve (0..15 m)": (0.0, 15.0), "exit (15..30 m)": (15.0, 30.0)}


def crossing_speed(s, t, lo, hi, end_censored):
    """Length over time to go from lo to hi along a monotone track, or NaN."""
    if s[0] > lo or s[-1] < lo:
        return np.nan
    t_in = np.interp(lo, s, t)
    if s[-1] >= hi:
        return (hi - lo) / max(np.interp(hi, s, t) - t_in, 1e-3)
    if not end_censored:          # cut by a collision inside the stretch
        return np.nan
    return (s[-1] - lo) / max(t[-1] - t_in, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with FEATURES.open() as f:
        curves = {r["clip_id"]: (float(r["turn_window_start_m"]), float(r["turn_window_end_m"]))
                  for r in csv.DictReader(f) if abs(float(r["max_windowed_yaw_deg"])) >= CURVE_DEG}
    files = usdz_paths()
    humans = {c: human_log(c, files.get(c, [USDZ / f"{c}.usdz"])) for c in curves}

    rows = []
    for clip, (ts, _) in curves.items():
        ht, hs, _ = humans[clip]
        hs_m, ht_s = np.maximum.accumulate(hs), (ht - ht[0]) / 1e6
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            rows.append({"model": "human", "clip": clip, "bin": lo,
                         "speed": crossing_speed(hs_m, ht_s, ts + lo, ts + hi, True)})
    for spec in a.run:
        label, run_dir = spec.split("=", 1)
        d = load_run(run_dir)
        d = d[d.clipgt_id.str.removeprefix("clipgt-").isin(curves)]
        for (cgt, _), g in d.groupby(["clipgt_id", "rollout_id"]):
            clip = cgt.removeprefix("clipgt-")
            ts = curves[clip][0]
            steps = g.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()
            t = (steps.index.values - humans[clip][0][0]) / 1e6
            s = np.maximum.accumulate(steps["dist_traveled_m"].values)
            rel = np.flatnonzero(steps["eval_relevant"].values > 0)
            hit = np.flatnonzero(steps["collision_any"].values > 0)
            i0, i1 = (rel[0] if rel.size else 0), (hit[0] if hit.size else len(t))
            t, s = t[i0:i1], s[i0:i1]
            if len(t) < 5:
                continue
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                rows.append({"model": label, "clip": clip, "bin": lo,
                             "speed": crossing_speed(s, t, ts + lo, ts + hi, not hit.size)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "curve_speed_profile.csv", index=False)

    # Every comparison uses, per bin, only clips where the human and every model
    # have a speed there, so all lines and the table describe the same clips.
    labels = [spec.split("=", 1)[0] for spec in a.run]
    drivers = ["human", *labels]
    prof = df.pivot_table(index=["clip", "bin"], columns="model", values="speed")[drivers].dropna()
    prof.to_csv(out / "curve_speed_profile_common.csv")

    # Stretch speed per clip = stretch length / time, over the common bins in it.
    lines = ["| stretch | clips | human | " + " | ".join(labels) + " | "
             + " | ".join(f"{m} < 50% of human" for m in labels) + " |",
             "|---|---|---|" + "---|" * (2 * len(labels))]
    for name, (lo, hi) in SEGMENTS.items():
        x = prof[(prof.index.get_level_values("bin") >= lo) & (prof.index.get_level_values("bin") < hi)]
        n_bins = x.groupby(level="clip").size()
        x = x[x.index.get_level_values("clip").isin(n_bins[n_bins >= 3].index)]
        seg = BIN_M * x.groupby(level="clip").size().to_frame("m")
        for m in drivers:
            seg[m] = seg["m"] / (BIN_M / x[m]).groupby(level="clip").sum()
        cells = [str(len(seg)), f"{seg['human'].median() * 3.6:.1f} km/h"]
        cells += [f"{seg[m].median() * 3.6:.1f} km/h ({(seg[m] / seg['human']).median():.2f})" for m in labels]
        cells += [f"{(seg[m] < 0.5 * seg['human']).sum()} ({(seg[m] < 0.5 * seg['human']).mean():.0%})" for m in labels]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "curve_speed_segments.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: median speed through each 2 m bin; one line per driver.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    colors = {"human": "black", labels[0]: "#d62728"}
    colors.update({m: c for m, c in zip(labels[1:], ["#1f77b4", "#2ca02c", "#9467bd"])})
    by_bin = {b: prof.xs(b, level="bin") for b in EDGES[:-1] if b in prof.index.get_level_values("bin")}
    centers = np.array([b + BIN_M / 2 for b in by_bin])
    for m in drivers:
        med = [x[m].median() * 3.6 for x in by_bin.values()]
        ax.plot(centers, med, lw=2.6 if m in ("human", labels[0]) else 1.6, color=colors[m],
                label="human (log)" if m == "human" else m)
    for m in labels:
        frac = [100 * (x[m] < 0.5 * x["human"]).mean() for x in by_bin.values()]
        ax2.plot(centers, frac, lw=2.2 if m == labels[0] else 1.4, color=colors[m], label=m)
    for axis in (ax, ax2):
        axis.axvspan(0, 15, color="0.9", zorder=0)
        axis.axvline(-20, color="0.6", lw=0.8, ls=":")
        axis.grid(alpha=0.3)
    ax.text(7.5, ax.get_ylim()[1] * 0.96, "curve", ha="center", va="top", color="0.35")
    ax.set_ylabel("median speed (km/h)")
    n = [len(x) for x in by_bin.values()]
    ax.set_title(f"Speed through curves (turn >= {CURVE_DEG:.0f} deg), same clips and route position per bin "
                 f"(n = {min(n)}-{max(n)} of {len(curves)})", fontsize=10.5)
    ax.legend(loc="lower left", frameon=False)
    ax2.set_ylabel("% of clips\n< 50% of human")
    ax2.set_xlabel("route position relative to the start of the curve (m)")
    fig.tight_layout()
    fig.savefig(out / "curve_speed_profile.png", dpi=150)
    print("wrote", out / "curve_speed_profile.png")


if __name__ == "__main__":
    main()
