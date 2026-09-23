"""What of a checkpoint actually lands in the model, and what does not.

`DriveSuprimAgent.initialize` loads with `strict=False`, which is what lets a
NAVSIM checkpoint trained without the route encoder load at all -- but it is
also silent. A renamed module, a changed width, a checkpoint for a different
backbone: all of them "load" and leave those weights at their random init, and
the only symptom is a run that trains worse than it should for no visible
reason.

This reports the three buckets by module, so the silence becomes a list:

  loaded       in both, shapes agree
  missing      the model has it, the checkpoint does not -> stays random
  unexpected   the checkpoint has it, this model does not -> discarded
  mismatched   in both, shapes differ -> discarded, stays random

Route weights showing up as `missing` is expected and fine when the checkpoint
predates route encoding: `_route_encoding` and `_route_tokens` are separate
additive modules, not a widening of `_status_encoding`, exactly so such a
checkpoint still loads.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.drivesuprim.drivesuprim_agent import DriveSuprimAgent

CFG_DIR = str(Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common/agent")


def _group(keys, depth: int = 2) -> dict:
    out = collections.Counter()
    for k in keys:
        out[".".join(k.split(".")[:depth])] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent", default="drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch")
    parser.add_argument("--use-route", action="store_true",
                        help="build the model with route encoding on (default: off)")
    parser.add_argument("--depth", type=int, default=2, help="module depth to group by")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if anything outside the route modules is missing")
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = instantiate(compose(config_name=args.agent)["config"])
    cfg.vocab_path = os.environ["NUREC_VOCAB_PATH"]
    cfg.use_route = args.use_route
    cfg.only_ori_input = True
    cfg.use_traffic_light_compliance = False
    cfg.training = False

    agent = DriveSuprimAgent(cfg, lr=1e-4)
    model_sd = agent.state_dict()

    raw = torch.load(args.checkpoint, map_location="cpu")
    ckpt_sd = raw.get("state_dict", raw)
    ckpt_sd = {k.replace("agent.", "", 1) if k.startswith("agent.") else k: v
               for k, v in ckpt_sd.items()}

    model_keys, ckpt_keys = set(model_sd), set(ckpt_sd)
    common = model_keys & ckpt_keys
    loaded = {k for k in common if tuple(model_sd[k].shape) == tuple(ckpt_sd[k].shape)}
    mismatched = common - loaded
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    n_model = sum(v.numel() for v in model_sd.values())
    n_loaded = sum(model_sd[k].numel() for k in loaded)
    print(f"checkpoint : {args.checkpoint}")
    print(f"agent      : {args.agent}   use_route={args.use_route}")
    print(f"model      : {len(model_keys):5d} tensors, {n_model/1e6:8.2f} M params")
    print(f"loaded     : {len(loaded):5d} tensors, {n_loaded/1e6:8.2f} M "
          f"({100*n_loaded/max(n_model,1):.1f}% of the model)")
    for label, keys in (("missing", missing), ("unexpected", unexpected),
                        ("mismatched", mismatched)):
        print(f"{label:11s}: {len(keys):5d} tensors")
        for mod, count in list(_group(keys, args.depth).items())[:12]:
            extra = ""
            if label == "mismatched":
                k = next(iter(k for k in keys if k.startswith(mod)))
                extra = f"   e.g. {k}: model {tuple(model_sd[k].shape)} vs ckpt {tuple(ckpt_sd[k].shape)}"
            print(f"    {mod:52s} {count:5d}{extra}")

    route_only = {k for k in missing if "route" in k.lower()}
    other = missing - route_only
    if route_only:
        print(f"\nroute modules missing (expected for a pre-route checkpoint): {len(route_only)}")
    if other:
        print(f"NOT explained by route: {len(other)} tensors")
        for mod, count in list(_group(other, args.depth).items())[:10]:
            print(f"    {mod:52s} {count:5d}")
    if args.strict and (other or mismatched):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
