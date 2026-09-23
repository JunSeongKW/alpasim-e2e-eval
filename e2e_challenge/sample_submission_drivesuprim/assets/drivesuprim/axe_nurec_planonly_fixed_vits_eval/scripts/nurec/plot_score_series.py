"""The scored checkpoints of the fixed plan-only run, against the old one.

score_epoch_series.sh scores every fifth checkpoint and appends a row to
exp/score_series/summary.tsv. This draws those rows with the old 30-epoch
miscalibrated run's final value as the line to beat.

Only one point exists for the old run -- it was scored once, at epoch 29 -- so
it is a horizontal rule rather than a curve. That is the honest shape of the
comparison: what the calibration bug cost is the gap between the curve and the
rule at the same epoch count, and by epoch 20 the curve is well clear of it.

  python scripts/nurec/plot_score_series.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "exp/score_series/summary.tsv"
OUT = ROOT / "exp/score_series/score_series.png"

# The old run, rescored under the corrected projection so the two numbers are
# produced by the same evaluation and differ only in the weights.
OLD = {"score": 0.9250, "NC": 0.9849, "DAC": 0.9682, "GT": 0.9966, "EP": 0.9706}
PANELS = [("score", "score  =  NC x DAC x GT x EP"), ("NC", "NC  no-at-fault collision"),
          ("DAC", "DAC  drivable area"), ("GT", "GT  4 m drift"), ("EP", "EP  ego progress")]


def main() -> None:
    rows = sorted(
        ({k: v for k, v in r.items()} for r in csv.DictReader(open(SUMMARY), delimiter="\t")),
        key=lambda r: int(r["epoch"]))
    if not rows:
        raise SystemExit(f"no rows in {SUMMARY}")
    ep = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.0 * len(PANELS), 3.6))
    for ax, (key, name) in zip(axes, PANELS):
        y = [float(r[key]) for r in rows]
        ax.plot(ep, y, color="#1f6f8b", lw=2.4, marker="o", ms=6, label="fixed run")
        ax.axhline(OLD[key], color="#b0413e", ls="--", lw=1.6,
                   label="old 30-epoch run")
        for x, v in zip(ep, y):
            ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#1f6f8b")
        best = max(y)
        ax.set_title(f"{name}\nbest {best:.4f}   old {OLD[key]:.4f}   "
                     f"{best - OLD[key]:+.4f}", fontsize=10,
                     color="#12603a" if best > OLD[key] else "#333333")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("plan-only, scored on the validation split (overlaps train — tracks the run, "
                 "does not qualify it)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight")

    print(f"{'epoch':>6s}" + "".join(f"{k:>10s}" for k, _ in PANELS))
    for r in rows:
        print(f"{int(r['epoch']):>6d}" + "".join(f"{float(r[k]):>10.4f}" for k, _ in PANELS))
    print(f"{'old':>6s}" + "".join(f"{OLD[k]:>10.4f}" for k, _ in PANELS))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
