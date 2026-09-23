"""Render an A/B comparison of the feasibility gate between two rollouts.

Both runs drive the same scene with the same fixed trajectory vocabulary, so a
candidate index means the same trajectory in both. That makes the gates directly
comparable per step: a candidate can survive in both, die in both, or flip. The
flips are what the ego-footprint change actually did, and they are invisible in
either run's own video because 4,096 candidates overplot into one blue smear.

Draws, per step:
  * candidates that flipped, in colour, over the unchanged ones in grey
  * both ego rectangles -- the box each gate believed it was clearing

Ego box sizes are passed in rather than read back, because run A predates the
API plumbing and reports nothing; its gate used the hardcoded default.
"""

import argparse
import asyncio
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from eval.asl_loader import load_scenario_eval_input_from_asl


def find_asl(run_dir: pathlib.Path) -> pathlib.Path:
    hits = sorted(run_dir.glob("rollouts/*/*/rollout.asl"))
    if not hits:
        raise SystemExit(f"no rollout.asl under {run_dir}")
    return hits[0]


def load(run_dir: pathlib.Path, cfg):
    asl = find_asl(run_dir)
    return asyncio.run(
        load_scenario_eval_input_from_asl(str(asl), cfg, {}, {"run_uuid": "cmp", "run_name": "cmp"})
    )


def masks_and_vocab(scenario):
    """Per-step feasibility masks plus the shared candidate vocabulary."""
    responses = scenario.driver_responses.per_timestep_driver_responses
    vocab = None
    masks = []
    for r in responses:
        if vocab is None and r.drivesuprim_candidate_vocab is not None:
            vocab = np.asarray(r.drivesuprim_candidate_vocab, dtype=np.float32)
        masks.append(
            None if r.drivesuprim_feasibility_mask is None
            else np.asarray(r.drivesuprim_feasibility_mask, dtype=bool)
        )
    return masks, vocab


def ego_rect(length: float, width: float, dx: float, **kw) -> Rectangle:
    # Plot axes are (lateral y, forward x); the box is axis-aligned because the
    # BEV is drawn in the un-rotated ego frame.
    return Rectangle(
        (-0.5 * width, dx - 0.5 * length), width, length,
        fill=False, linewidth=1.8, **kw,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--a-box", default="4.6,1.9,0.0", help="length,width,dx of run A's gate box")
    p.add_argument("--b-box", default="5.207,2.157,1.467", help="same for run B")
    p.add_argument("--label-a", default="ego box A: 4.60 x 1.90, offset 0")
    p.add_argument("--label-b", default="ego box B: 5.207 x 2.157, offset +1.467 m")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--radius", type=float, default=32.0)
    args = p.parse_args()

    a_box = tuple(float(v) for v in args.a_box.split(","))
    b_box = tuple(float(v) for v in args.b_box.split(","))

    # EvalConfig has mandatory fields, so start from a run's own resolved config
    # rather than the bare schema.
    cfg_path = pathlib.Path(args.run_a) / "eval-config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}; pass a run directory produced by the wizard")
    # Use the run's own eval-config.yaml as-is rather than merging it into the
    # EvalConfig schema: the schema declares mandatory fields the wizard fills in
    # at runtime and leaves out of the file, and to_object refuses to build an
    # object while any of them is MISSING. Nothing on this path reads them, and
    # attribute access works the same on a plain DictConfig.
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.parse_unstructured_debug_info = True
    cfg.video.render_video = False

    print("loading run A ...", flush=True)
    a = load(pathlib.Path(args.run_a), cfg)
    print("loading run B ...", flush=True)
    b = load(pathlib.Path(args.run_b), cfg)

    masks_a, vocab = masks_and_vocab(a)
    masks_b, vocab_b = masks_and_vocab(b)
    if vocab is None:
        vocab = vocab_b
    if vocab is None:
        raise SystemExit("neither run carries a candidate vocabulary; rerun with debug parsing on")

    steps = min(len(masks_a), len(masks_b))
    print(f"steps: {steps}, vocab: {vocab.shape}", flush=True)

    # Candidate polylines in plot coords (lateral, forward).
    segs = [np.column_stack([t[:, 1], t[:, 0]]) for t in vocab]

    fig, ax = plt.subplots(figsize=(7.5, 8.0), dpi=110)
    ax.set_xlim(args.radius, -args.radius)
    ax.set_ylim(-0.25 * args.radius, 1.75 * args.radius)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, color="white", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("left  <-  lateral y (m)  ->  right", fontsize=8)
    ax.set_ylabel("forward x (m)", fontsize=8)

    same = LineCollection([], colors="#c9c9c9", linewidths=0.4, alpha=0.22, zorder=1)
    only_b_kills = LineCollection([], colors="#e11d48", linewidths=0.9, alpha=0.75, zorder=3)
    only_a_kills = LineCollection([], colors="#2563eb", linewidths=0.9, alpha=0.75, zorder=3)
    for c in (same, only_b_kills, only_a_kills):
        ax.add_collection(c)

    rect_a = ego_rect(*a_box, edgecolor="#1f2937", linestyle=(0, (4, 3)), zorder=6)
    rect_b = ego_rect(*b_box, edgecolor="#15803d", zorder=6)
    ax.add_patch(rect_a)
    ax.add_patch(rect_b)

    title = ax.set_title("", fontsize=10)
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#e11d48", lw=2, label="killed by B only"),
            plt.Line2D([], [], color="#2563eb", lw=2, label="killed by A only"),
            plt.Line2D([], [], color="#c9c9c9", lw=2, label="same verdict in both"),
            plt.Line2D([], [], color="#1f2937", lw=2, ls="--", label=args.label_a),
            plt.Line2D([], [], color="#15803d", lw=2, label=args.label_b),
        ],
        loc="lower right", fontsize=7, framealpha=0.9,
    )
    fig.tight_layout()

    def update(i):
        ma, mb = masks_a[i], masks_b[i]
        if ma is None or mb is None:
            same.set_segments(segs)
            only_b_kills.set_segments([])
            only_a_kills.set_segments([])
            title.set_text(f"step {i + 1}/{steps}  (no gate data)")
            return same, only_b_kills, only_a_kills, title
        b_extra = ma & ~mb          # survived A, killed by B
        a_extra = ~ma & mb          # killed by A, survived B
        unchanged = ~(b_extra | a_extra)
        same.set_segments([segs[j] for j in np.flatnonzero(unchanged)])
        only_b_kills.set_segments([segs[j] for j in np.flatnonzero(b_extra)])
        only_a_kills.set_segments([segs[j] for j in np.flatnonzero(a_extra)])
        title.set_text(
            f"step {i + 1}/{steps}   |   survived  A {int(ma.sum()):,}  B {int(mb.sum()):,}"
            f"   |   flips  +{int(b_extra.sum())} / -{int(a_extra.sum())}"
        )
        return same, only_b_kills, only_a_kills, title

    # ffmpeg lives in the AlpaSim container, not on the host, so write numbered
    # PNGs and let the caller encode them.
    out = pathlib.Path(args.out)
    frames_dir = out.parent / (out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("f*.png"):
        old.unlink()
    for i in range(steps):
        update(i)
        fig.savefig(frames_dir / f"f{i:04d}.png")
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{steps}", flush=True)
    print(f"frames -> {frames_dir}", flush=True)

    # A one-line summary is the part worth quoting; the video only shows it moving.
    tot_b = sum(int((ma & ~mb).sum()) for ma, mb in zip(masks_a, masks_b) if ma is not None and mb is not None)
    tot_a = sum(int((~ma & mb).sum()) for ma, mb in zip(masks_a, masks_b) if ma is not None and mb is not None)
    print(f"total flips over {steps} steps:  B killed {tot_b:,} more,  A killed {tot_a:,} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
