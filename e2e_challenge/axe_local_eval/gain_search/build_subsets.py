"""Pick the clip subsets the controller-gain search runs on.

A proportional sample wastes half its budget: 48.5% of curated_val clips score
a clean 1.0 under the current gains and almost never move, so most rollouts
buy no information about which gains are better. Instead the clips are
stratified by how the reference run behaved on them and sampled with Neyman
allocation -- heavily under-sampling the easy stratum and over-sampling the
ones that actually vary -- then each stratum is weighted back to its
population share when scoring. That estimates the same full-set mean with a
fraction of the rollouts.

The easy stratum keeps a floor of 30 clips rather than dropping out entirely,
because a gain setting that breaks scenes the current one handles cleanly is
exactly the failure this search must not miss.

A disjoint holdout is drawn the same way. The search will be run many times on
SEARCH and will overfit it; HOLDOUT is what a candidate has to survive before
it is worth a full 441-clip run.
"""

import collections
import json
import pathlib
import random

RUN = pathlib.Path(
    "/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/axe-ep24-curatedval-egobox"
)
OUT = pathlib.Path(__file__).parent

# Rollouts per clip in the reference run; a clip is classified from all of them.
ALLOC = {"perfect": 30, "partial": 35, "always_fail": 25, "mixed": 30}
SEED = 20260912


def classify(rollouts):
    reasons = [r for _, r in rollouts]
    mean = sum(s for s, _ in rollouts) / len(rollouts)
    if all(r == "pass" for r in reasons) and mean >= 0.999:
        return "perfect"
    if all(r != "pass" for r in reasons):
        return "always_fail"
    if any(r != "pass" for r in reasons):
        return "mixed"
    return "partial"


summary = json.loads((RUN / "aggregate" / "results-summary.json").read_text())
clips = collections.defaultdict(list)
for r in summary["rollouts"]:
    clips[r["clipgt_id"]].append((r["score"], r.get("failure_reason") or "pass"))

strata = collections.defaultdict(list)
for clip, rollouts in clips.items():
    strata[classify(rollouts)].append(clip)

total = sum(len(v) for v in strata.values())
rng = random.Random(SEED)

search, holdout, weights = {}, {}, {}
print(f"{'stratum':14s} {'pop':>5s} {'share':>7s} {'search':>7s} {'holdout':>8s} {'weight':>8s}")
for name in sorted(strata, key=lambda k: -len(strata[k])):
    pool = sorted(strata[name])
    rng.shuffle(pool)
    n = min(ALLOC[name], len(pool) // 2)
    search[name], holdout[name] = pool[:n], pool[n : 2 * n]
    weights[name] = len(pool) / total
    print(f"{name:14s} {len(pool):5d} {len(pool)/total:6.1%} {n:7d} {n:8d} {weights[name]:8.4f}")

flat_s = [c for v in search.values() for c in v]
flat_h = [c for v in holdout.values() for c in v]
assert not (set(flat_s) & set(flat_h))
print(f"\nSEARCH {len(flat_s)} clip   HOLDOUT {len(flat_h)} clip   overlap 0")

meta = {
    "source_run": RUN.name,
    "seed": SEED,
    "stratum_weights": weights,
    "population": {k: len(v) for k, v in strata.items()},
    "search": search,
    "holdout": holdout,
}
(OUT / "subsets.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

for label, groups in (("search", search), ("holdout", holdout)):
    ids = sorted(c for v in groups.values() for c in v)
    (OUT / f"{label}_clips.txt").write_text("\n".join(ids) + "\n")
    # Hydra list override; the wizard takes scenes.scene_ids as a list.
    (OUT / f"{label}_scene_ids.txt").write_text("[" + ",".join(ids) + "]")

# Sanity: the weighted estimator must reproduce the reference run's full mean.
for label, groups in (("SEARCH", search), ("HOLDOUT", holdout)):
    est = sum(
        weights[name]
        * (sum(s for c in ids for s, _ in clips[c]) / sum(len(clips[c]) for c in ids))
        for name, ids in groups.items()
    )
    naive = [s for ids in groups.values() for c in ids for s, _ in clips[c]]
    print(f"{label:8s} weighted {est:.4f}   unweighted {sum(naive)/len(naive):.4f}")
full = [s for v in clips.values() for s, _ in v]
print(f"{'FULL':8s} {sum(full)/len(full):.4f}  (441 clip x 3)")
