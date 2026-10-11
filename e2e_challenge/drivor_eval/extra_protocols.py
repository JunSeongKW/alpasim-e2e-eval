# SPDX-License-Identifier: Apache-2.0
"""More AlpaSim metric conventions from the literature, on the same rollouts.

    .venv/bin/python e2e_challenge/drivor_eval/extra_protocols.py \
        --run ours=runs/drivor-ours-hugsim-final-441x3-20261009 \
        --run original=runs/drivor-drivor-original-441x3-20261009 --out runs/drivor-protocols-20261009

rescore_protocols.py covers the challenge, Alpamayo-R1 and RoaD (App. B.1)
conventions. This adds three more, each computed from the per-timestep metrics
of every rollout, from the first eval_relevant step on:

  raw        FlashDrive (arXiv 2608.12932, Table A3): every metric reduced with
             AlpaSim's own time aggregation (max for events, min for progress_rel,
             max for distances, mean for plan deviation) over the whole rollout,
             with no truncation at an event and no 4 m rule.
  omnidreams OmniDreams (arXiv 2606.03159, Sec. 9.4.1): "an incident such as a
             collision or off-road event is only counted when the ego is within
             4 m of the corresponding ground-truth trajectory"; collision split
             into front / lateral / rear. Each rollout's first counted contact is
             classed by its first frame (front, else lateral, else rear), so the
             three parts add up to the collision rate.
  road_main  RoaD main text (arXiv 2512.01993, Sec. 5.2.1), which also excludes
             side contacts ("excluding rear-end and side contacts"); App. B.1
             keeps lateral contacts, and that variant is the "road" table of
             rescore_protocols.py. Incident = front contact outside a rear-started
             contact episode, or off-road; count stops at the first incident and
             at 4 m from the log.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd

from rescore_protocols import CORRIDOR_M, at_fault_outside_rear_episodes


def steps(path: Path) -> tuple:
    d = pd.read_parquet(path, columns=["name", "timestamps_us", "values", "time_aggregation"])
    agg = d.drop_duplicates("name").set_index("name")["time_aggregation"].to_dict()
    w = d.pivot_table(index="timestamps_us", columns="name", values="values", aggfunc="first").sort_index()
    rel = w.index[w["eval_relevant"] > 0]
    return (w.loc[w.index >= rel[0]] if len(rel) else w), agg


def raw(w, agg):
    out = {}
    for name in ("collision_any", "collision_front", "collision_lateral", "collision_rear", "offroad", "wrong_lane",
                 "progress_rel", "dist_to_gt_trajectory", "dist_to_gt_location", "plan_deviation"):
        x = w[name].dropna()
        if not len(x):
            out[name] = math.nan
        else:
            out[name] = float(x.iloc[-1] if agg[name] == "last" else getattr(x, agg[name])())
    return out


def omnidreams(w):
    near = w["dist_to_gt_trajectory"] < CORRIDOR_M
    contact = (w["collision_any"] > 0) & near
    first = w.index[contact]
    kind = None
    if len(first):
        row = w.loc[first[0]]
        kind = "front" if row["collision_front"] > 0 else "lateral" if row["collision_lateral"] > 0 else "rear"
    offroad = bool(((w["offroad"] > 0) & near).any())
    return {"collision": float(kind is not None), "front": float(kind == "front"), "lateral": float(kind == "lateral"),
            "rear": float(kind == "rear"), "offroad": float(offroad), "incident": float(kind is not None or offroad)}


def road_main(w):
    def cut(mask):
        nonlocal w
        hit = w.index[mask.reindex(w.index).fillna(False).astype(bool)]
        if len(hit):
            w = w.loc[w.index <= hit[0]]

    cut(w["dist_to_gt_trajectory"] >= CORRIDOR_M)
    front = at_fault_outside_rear_episodes(w) & (w["collision_front"] > 0)
    cut(front | (w["offroad"] > 0))
    front = at_fault_outside_rear_episodes(w) & (w["collision_front"] > 0)
    return {"collision": float(front.any()), "offroad": float((w["offroad"] > 0).any()),
            "dist_m": float(w["dist_traveled_m"].iloc[-1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tables = {"raw": {}, "omnidreams": {}, "road_main": {}}
    for item in a.run:
        name, run = item.split("=", 1)
        rows = {"raw": [], "omnidreams": [], "road_main": []}
        for m in sorted(Path(run).glob("rollouts/*/*/metrics.parquet")):
            if not (m.parent / "_complete").exists():
                continue
            w, agg = steps(m)
            rows["raw"].append(raw(w, agg))
            rows["omnidreams"].append(omnidreams(w))
            rows["road_main"].append(road_main(w))
        n = len(rows["raw"])
        mean = lambda rs, k, s=100: s * statistics.fmean(r[k] for r in rs)  # noqa: E731
        tables["raw"][name] = {"rollouts": n, **{k: mean(rows["raw"], k, 100 if k in (
            "collision_any", "collision_front", "collision_lateral", "collision_rear", "offroad", "wrong_lane") else 1)
            for k in rows["raw"][0]}}
        tables["omnidreams"][name] = {"rollouts": n, **{k: mean(rows["omnidreams"], k) for k in rows["omnidreams"][0]}}
        r = rows["road_main"]
        inc = sum(1 for x in r if x["collision"] or x["offroad"])
        tables["road_main"][name] = {"rollouts": n, "collision": mean(r, "collision"), "offroad": mean(r, "offroad"),
                                     "dist_m": mean(r, "dist_m", 1),
                                     "km_per_incident": sum(x["dist_m"] for x in r) / 1000 / inc if inc else math.inf}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "extra_protocols.json").write_text(json.dumps(tables, indent=1) + "\n")
    print(json.dumps(tables, indent=1))


if __name__ == "__main__":
    main()
