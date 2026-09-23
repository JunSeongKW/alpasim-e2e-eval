"""Validation loss of the running plan-only job against the miscalibrated one.

The two runs share the split, the labels, the head set and the LR schedule --
train_logs and val_logs are the same 1552 / 458 clips, pdm_heads is the same
four, and both follow scheduler_epoch 30 with the same warmup. The one
difference that matters is the camera projection: the old run's logs carried
nuPlan's 1920x1080 pinhole against rectified 512x256 files, so no reference
point landed inside an image and the BEV carried no camera information at all.

That makes the pair a clean read on what the calibration was worth, term by
term. NC and DAC need vision; GT is ego-trajectory geometry and should not care.

  python scripts/nurec/plot_val_compare.py [NEW_RUN_DIR] [OLD_RUN_DIR]

Prints a table and writes exp/bev_feature_compare/val_loss_compare.png.
Exits 2 if the new run has no completed validation epoch yet.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
NEW_EXP = ROOT / "exp/nurec_vits_nurec_planonly"
OLD_RUN = ROOT / "exp/nurec_pdms_v18_vits_planonly_bf16/2026.08.29.05.07.45"
OUT = ROOT / "exp/bev_feature_compare/val_loss_compare.png"

# Every val/*_epoch scalar gets a panel. The order below is the reading order --
# the ori branch first (the one the score is built from), then the soft branch
# the SSL teacher supervises, then the totals -- and anything not named here is
# appended rather than dropped, so a stage that adds heads shows up on its own.
NAMES = [
    ("val/pdm_aggregate_loss-ori_epoch", "aggregate  NC x DAC x GT x EP"),
    ("val/pdm_noc_loss-ori_epoch", "NC  no-at-fault collision"),
    ("val/pdm_da_loss-ori_epoch", "DAC  drivable area"),
    ("val/pdm_gt_loss-ori_epoch", "GT  4 m drift"),
    ("val/pdm_progress_loss-ori_epoch", "EP  ego progress"),
    ("val/imi_loss-ori_epoch", "imi  imitation"),
    ("val/loss-ori_epoch", "total (ori branch)"),
    ("val/loss-refinement_ori_epoch", "refinement head"),
    ("val/stage_2_layer_1-ori_epoch", "refinement layer 1"),
    ("val/stage_2_layer_2-ori_epoch", "refinement layer 2"),
    ("val/stage_2_layer_3-ori_epoch", "refinement layer 3"),
    ("val/pdm_aggregate_loss-soft_epoch", "aggregate (soft)"),
    ("val/pdm_noc_loss-soft_epoch", "NC (soft)"),
    ("val/pdm_da_loss-soft_epoch", "DAC (soft)"),
    ("val/pdm_gt_loss-soft_epoch", "GT (soft)"),
    ("val/pdm_progress_loss-soft_epoch", "EP (soft)"),
    ("val/imi_loss-soft_epoch", "imi (soft)"),
    ("val/loss-soft_epoch", "total (soft branch)"),
    ("val/loss_epoch", "grand total"),
]

# Terms the calibration should move, and terms it should not. NC and DAC cannot
# be learned without seeing the scene; GT is ego-trajectory geometry and reads
# the same either way. Marking them makes the expected pattern checkable rather
# than something to eyeball.
VISION_BOUND = {"val/pdm_noc_loss-ori_epoch", "val/pdm_da_loss-ori_epoch",
                "val/pdm_noc_loss-soft_epoch", "val/pdm_da_loss-soft_epoch"}


def panels(old_tags, new_tags) -> List[tuple]:
    have = set(old_tags) & set(new_tags)
    named = [(t, n) for t, n in NAMES if t in have]
    rest = sorted(t for t in have if t not in {t for t, _ in NAMES})
    return named + [(t, t[len("val/"):-len("_epoch")]) for t in rest]


def latest_new_run() -> Path:
    runs = sorted(p for p in NEW_EXP.glob("2*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run under {NEW_EXP}")
    return runs[-1]


_ACC = {}


def acc(run: Path):
    """One EventAccumulator per run.

    Reload() parses the whole event file, and the old run's is 17 MB. Reading it
    once per tag turned a two-second plot into a two-minute one.
    """
    key = str(run)
    if key not in _ACC:
        ev = glob.glob(f"{run}/lightning_logs/version_0/events.out.tfevents*")
        if not ev:
            _ACC[key] = None
        else:
            ea = EventAccumulator(ev[0], size_guidance={"scalars": 0})
            ea.Reload()
            _ACC[key] = ea
    return _ACC[key]


def all_tags(run: Path) -> List[str]:
    ea = acc(run)
    if ea is None:
        return []
    return [t for t in ea.Tags()["scalars"]
            if t.startswith("val/") and t.endswith("_epoch")]


def series(run: Path, tag: str) -> List[float]:
    ea = acc(run)
    if ea is None or tag not in ea.Tags()["scalars"]:
        return []
    return [e.value for e in ea.Scalars(tag)]


def main() -> None:
    new_run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_new_run()
    old_run = Path(sys.argv[2]) if len(sys.argv) > 2 else OLD_RUN
    if not series(new_run, "val/loss-ori_epoch"):
        print(f"{new_run.name}: no validation epoch finished yet")
        raise SystemExit(2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    todo = panels(all_tags(old_run), all_tags(new_run))
    ncol = 5
    nrow = -(-len(todo) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow), squeeze=False)
    for ax in axes.ravel()[len(todo):]:
        ax.axis("off")
    rows = []
    for ax, (tag, name) in zip(axes.ravel(), todo):
        o, n = series(old_run, tag), series(new_run, tag)
        ax.plot(range(len(o)), o, color="#b0413e", lw=1.5, marker="o", ms=2.2)
        ax.plot(range(len(n)), n, color="#1f6f8b", lw=2.3, marker="o", ms=4.5)
        ax.axhline(o[-1], color="#b0413e", ls=":", lw=1.0)
        beaten = min(n) < o[-1]
        mark = " *" if tag in VISION_BOUND else ""
        ax.set_title(f"{name}{mark}\n{'PAST old final' if beaten else f'old final {o[-1]:.4f}'}",
                     fontsize=9.5, color="#12603a" if beaten else "#333333")
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25)
        # The epoch at which the old run first reached where the new one is now
        match = next((i for i, v in enumerate(o) if v <= n[-1]), None)
        rows.append((name, n[-1], len(n) - 1, o[-1], min(o),
                     str(match) if match is not None else f"never ({len(o)} ep)"))

    ep = rows[0][2]
    fig.suptitle("plan-only validation, every logged term   |   "
                 f"red = old 30-epoch miscalibrated run, blue = fixed run (epoch {ep})   |   "
                 "same split, labels and schedule; only the camera calibration differs   |   "
                 "* = cannot be learned without vision", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=115, bbox_inches="tight")

    print(f"new run: {new_run.name}   epochs validated: {rows[0][2] + 1}")
    print(f"{'term':34s}{'new now':>10s}{'old final':>11s}"
          f"{'old best':>10s}{'old ep to match':>17s}")
    for name, cur, _ep, ofin, obest, match in rows:
        print(f"{name:34s}{cur:>10.4f}{ofin:>11.4f}{obest:>10.4f}{match:>17s}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
