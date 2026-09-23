"""Decide which controller-gain set to evaluate next.

The search is deliberately model-free. Each evaluation costs about half an hour
of wall clock, so with a few dozen points in a 7-dimensional box there is not
enough data to fit a surrogate that would beat plain coordinate search, and a
surrogate's extrapolations would be spent on runs that cannot be taken back.

Four phases, in order, each falling through to the next when it has nothing
left to propose:

  screen   One-factor-at-a-time from the anchor, round-robin across parameters
           and nearest-value-first within each. After one lap the local slope in
           every direction is known, which is what decides where refinement is
           worth spending hours.

  confirm  Re-measure the incumbent until it has REPEATS measurements. A single
           run carries SE ~0.026 while the effects being chased are 0.01-0.03,
           so a greedy climb on single measurements would mostly chase noise.
           Averaging the leader is the cheapest way to stop that.

  refine   Coordinate descent from the incumbent over the same grid.

  explore  Random multi-axis perturbations around the incumbent, log-scaled and
           clipped to the competition bounds. This phase never runs out, which
           is what lets the loop run until someone stops it.

Everything is derived from results.jsonl, so the loop is restartable: kill it
mid-run and the next invocation proposes the same point. Scores for a repeated
gain set are averaged, so the incumbent is chosen on means rather than on a
single lucky run.

The grid and every perturbation stay inside the bounds the competition CLI
enforces (CONTROLLER_GAIN_BOUNDS, IDX_START_PENALTY_BOUNDS); a gain set that
cannot be submitted is not worth measuring.
"""

import argparse
import collections
import json
import os
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).parent
# One results file per model. The incumbent, the screen plan and the explore
# seed are all derived from this file, so pointing two different checkpoints at
# the same one would have each model's runs chosen as the other's starting
# point. GAIN_ANCHOR moves the origin of the search with it.
RESULTS = pathlib.Path(os.environ.get("GAIN_RESULTS") or HERE / "results.jsonl")

BOUNDS = {
    "long_position_weight": (0.0, 10.0),
    "lat_position_weight": (0.0, 10.0),
    "heading_weight": (0.0, 10.0),
    "acceleration_weight": (0.0, 10.0),
    "rel_front_steering_angle_weight": (0.0, 10.0),
    "rel_acceleration_weight": (0.0, 10.0),
    "idx_start_penalty": (0, 19),
}

# The gains the two completed 441-clip runs used. GAIN_ANCHOR overrides this
# with a JSON object when a search starts from a checkpoint that already has a
# tuned set -- the screen phase measures deviations from the anchor, so leaving
# it at the stock values would spend the first lap re-deriving what is known.
ANCHOR = {
    "long_position_weight": 0.5,
    "lat_position_weight": 6.0,
    "heading_weight": 1.0,
    "acceleration_weight": 0.1,
    "rel_front_steering_angle_weight": 5.0,
    "rel_acceleration_weight": 1.0,
    "idx_start_penalty": 3,
}
if os.environ.get("GAIN_ANCHOR"):
    ANCHOR = {**ANCHOR, **json.loads(os.environ["GAIN_ANCHOR"])}

GRID = {
    "long_position_weight": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "lat_position_weight": [1.0, 3.0, 6.0, 8.0, 10.0],
    "heading_weight": [0.0, 0.3, 1.0, 3.0, 6.0, 10.0],
    "acceleration_weight": [0.0, 0.1, 0.5, 2.0],
    "rel_front_steering_angle_weight": [1.0, 2.5, 5.0, 10.0],
    "rel_acceleration_weight": [0.2, 1.0, 3.0, 8.0],
    "idx_start_penalty": [0, 1, 6, 10, 15],
}

# Screened in this order: the three already tuned for the submission first,
# since they are known to move the score, then the untouched regularisers.
ORDER = [
    "lat_position_weight",
    "long_position_weight",
    "idx_start_penalty",
    "heading_weight",
    "rel_front_steering_angle_weight",
    "rel_acceleration_weight",
    "acceleration_weight",
]

SHORT = {
    "long_position_weight": "lon",
    "lat_position_weight": "lat",
    "heading_weight": "hdg",
    "acceleration_weight": "acc",
    "rel_front_steering_angle_weight": "rst",
    "rel_acceleration_weight": "rac",
    "idx_start_penalty": "idx",
}

REPEATS = 3          # measurements to average before trusting the incumbent
SEED = 20260912


def load():
    if not RESULTS.exists():
        return []
    out = []
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def key(gains) -> str:
    return json.dumps({k: gains[k] for k in sorted(BOUNDS)}, sort_keys=True)


def grouped(done):
    """Unique gain set -> {gains, scores, mean, n, tag}."""
    g = {}
    for r in done:
        if r.get("score") is None:
            continue
        k = key(r["gains"])
        e = g.setdefault(k, {"gains": r["gains"], "scores": [], "tag": r["tag"]})
        e["scores"].append(r["score"])
    for e in g.values():
        e["n"] = len(e["scores"])
        e["mean"] = statistics.fmean(e["scores"])
    return g


def best_point(done):
    g = grouped(done)
    if not g:
        return dict(ANCHOR), None
    top = max(g.values(), key=lambda e: e["mean"])
    return dict(top["gains"]), top


def distance(param, value):
    """Nearest-first ordering, on a log scale so 0.25 and 1.0 are equidistant from 0.5."""
    a = ANCHOR[param]
    lo, hi = min(a, value), max(a, value)
    return (hi + 0.05) / (lo + 0.05)


def screen_plan():
    """OFAT deviations, round-robin across parameters, nearest value first."""
    per_param = {
        p: sorted((v for v in GRID[p] if v != ANCHOR[p]), key=lambda v: distance(p, v))
        for p in ORDER
    }
    plan, lap = [], 0
    while any(len(v) > lap for v in per_param.values()):
        for p in ORDER:
            if len(per_param[p]) > lap:
                plan.append((p, per_param[p][lap]))
        lap += 1
    return plan


def with_value(base, param, value):
    g = dict(base)
    g[param] = value
    return g


def clip(param, value):
    lo, hi = BOUNDS[param]
    value = max(lo, min(hi, value))
    if param == "idx_start_penalty":
        return int(round(value))
    return round(value, 3)


def perturb(base, rng):
    """Multiplicative jitter on 1-3 axes, clipped to the submittable box."""
    g = dict(base)
    for param in rng.sample(ORDER, rng.randint(1, 3)):
        lo, hi = BOUNDS[param]
        cur = float(g[param])
        if cur <= 1e-9:
            # A zeroed weight cannot be scaled back up; step in from the floor.
            g[param] = clip(param, rng.uniform(0.0, 0.25 * hi))
            continue
        g[param] = clip(param, cur * rng.lognormvariate(0.0, rng.choice([0.25, 0.5, 0.9])))
    return g


def propose(done):
    # A config whose run failed has no score, and failures are usually
    # environmental (a full disk took out 52 runs once). Treat it as untried so
    # it gets another chance, but give up after two failures so a genuinely
    # broken config cannot stall the search.
    failures = collections.Counter(key(r["gains"]) for r in done if r.get("score") is None)
    seen = {key(r["gains"]) for r in done if r.get("score") is not None}
    seen |= {k for k, n in failures.items() if n >= 2}

    if key(ANCHOR) not in seen:
        return ANCHOR, "anchor", "baseline gains, same as the 441-clip runs"

    for param, value in screen_plan():
        g = with_value(ANCHOR, param, value)
        if key(g) not in seen:
            return (
                g,
                f"screen-{SHORT[param]}{value:g}",
                f"OFAT: {param} {ANCHOR[param]:g} -> {value:g}",
            )

    base, top = best_point(done)
    # A run that fails does not raise n, so confirming a config whose runs keep
    # failing would propose it forever. Two failures is enough to move on.
    if top is not None and top["n"] < REPEATS and failures[key(top["gains"])] < 2:
        return (
            dict(top["gains"]),
            f"confirm-{top['tag'].split('-', 1)[-1]}",
            f"repeat {top['tag']} to average out noise "
            f"(measurement {top['n'] + 1} of {REPEATS}, mean so far {top['mean']:.4f})",
        )

    for param in ORDER:
        vals = GRID[param]
        here = vals.index(base[param]) if base[param] in vals else len(vals) // 2
        for value in sorted(vals, key=lambda v: abs(vals.index(v) - here)):
            if value == base[param]:
                continue
            g = with_value(base, param, value)
            if key(g) not in seen:
                return (
                    g,
                    f"refine-{SHORT[param]}{value:g}",
                    f"coordinate descent from {top['tag']} ({top['mean']:.4f}): "
                    f"{param} {base[param]:g} -> {value:g}",
                )

    # Unbounded phase: keeps proposing until someone stops the loop. Seeded by
    # how many runs exist so a restart re-proposes the same point.
    rng = random.Random(SEED + len(done))
    for _ in range(200):
        g = perturb(base, rng)
        if key(g) not in seen:
            changed = [f"{SHORT[p]} {base[p]:g}->{g[p]:g}" for p in ORDER if g[p] != base[p]]
            return (
                g,
                f"explore-{len(done):03d}",
                f"random local search from {top['tag']} ({top['mean']:.4f}): "
                + ", ".join(changed),
            )
    # Every neighbour tried: widen by perturbing the anchor instead.
    g = perturb(ANCHOR, rng)
    return g, f"explore-wide-{len(done):03d}", "random search from the anchor"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("next", "status"))
    args = p.parse_args()

    done = load()

    if args.command == "status":
        g = grouped(done)
        rows = sorted(g.values(), key=lambda e: -e["mean"])
        print(
            f"{'rank':>4s} {'tag':22s} {'mean':>8s} {'n':>2s} {'runs':>22s}  gains"
        )
        for i, e in enumerate(rows, 1):
            gg = e["gains"]
            gs = " ".join(f"{SHORT[k]}={gg[k]:g}" for k in ORDER)
            runs = " ".join(f"{s:.3f}" for s in e["scores"])
            print(f"{i:4d} {e['tag']:22s} {e['mean']:8.4f} {e['n']:2d} {runs:>22s}  {gs}")
        failed = sum(1 for r in done if r.get("score") is None)
        print(f"\n{len(done)} runs, {len(g)} unique gain sets" + (f", {failed} failed" if failed else ""))
        return 0

    gains, tag, why = propose(done)
    print(json.dumps({"tag": f"{len(done):03d}-{tag}", "gains": gains, "why": why}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
