#!/usr/bin/env python3
"""Write one ranking-config variant for the sweep.

  make_rank_variant.py NAME [--coarse-exp NC:1,DAC:3] [--fine-exp DAC:5]
                            [--coarse-imi 1.0] [--fine-imi 0.02]

Omitted knobs keep the baseline's value, so a variant differs from the baseline
only in what is named on the command line.
"""
import argparse, copy, json, pathlib

PKG = pathlib.Path(__file__).parent / "assets/drivesuprim/stage3_ep24_eval"
BASE = PKG / "stage3_config.json"
OUT = PKG / "rank_variants"
KEY = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "EP": "ego_progress"}


def parse_exp(s):
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        k = k.strip().upper()
        out[KEY.get(k, k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--coarse-exp", default=None)
    ap.add_argument("--fine-exp", default=None)
    ap.add_argument("--coarse-imi", type=float, default=1.0)
    ap.add_argument("--fine-imi", type=float, default=0.02)
    a = ap.parse_args()

    c = copy.deepcopy(json.load(open(BASE)))
    c["pdm_imi_rank_weight"] = a.coarse_imi
    c["pdm_imi_rank_weight_refine"] = a.fine_imi
    c["pdm_rank_product_exponents"] = parse_exp(a.coarse_exp) or {}
    fine = parse_exp(a.fine_exp)
    if fine is not None:
        c["pdm_rank_product_exponents_refine"] = fine

    OUT.mkdir(exist_ok=True)
    p = OUT / f"{a.name}.json"
    json.dump(c, open(p, "w"), indent=2, sort_keys=True)
    print(f"{p}")
    print(f"  coarse  exp={c['pdm_rank_product_exponents']}  imi={c['pdm_imi_rank_weight']}")
    print(f"  fine    exp={c.get('pdm_rank_product_exponents_refine', '(coarse 와 동일)')}"
          f"  imi={c['pdm_imi_rank_weight_refine']}")


if __name__ == "__main__":
    main()
