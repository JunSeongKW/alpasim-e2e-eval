"""NaN/Inf source locator.

Registers forward hooks on every submodule. When a module's OUTPUT is the first
to become non-finite while its INPUTS are still finite, that module is the true
source (an overflow/log(0)/etc. inside it, or a corrupted buffer/param). We print
exactly which module + whether its buffers/params are non-finite, then raise so
the stack trace points at it.

Enable by setting env var BEVFORMER_NAN_DEBUG=1 before training.
"""
import os
import torch


def _any_nonfinite(x):
    # Detect actual NaN only. We intentionally do NOT flag +/-inf: this codebase
    # legitimately produces -inf via log() of near-zero probabilities in the
    # scoring path, which would otherwise be false positives. NaN is the real
    # corruption signal we are hunting.
    if isinstance(x, torch.Tensor):
        return x.is_floating_point() and torch.isnan(x).any()
    if isinstance(x, (list, tuple)):
        return any(_any_nonfinite(i) for i in x)
    if isinstance(x, dict):
        return any(_any_nonfinite(i) for i in x.values())
    return False


def register_nan_hooks(model):
    if os.environ.get("BEVFORMER_NAN_DEBUG", "") not in ("1", "true", "True"):
        return
    print("[NAN-DEBUG] hooks registered on all submodules", flush=True)

    def make_hook(name, module):
        def hook(mod, inp, out):
            if not _any_nonfinite(out):
                return
            in_bad = _any_nonfinite(inp)
            buf_bad = [n for n, b in mod.named_buffers(recurse=False)
                       if b.is_floating_point() and not torch.isfinite(b).all()]
            par_bad = [n for n, p in mod.named_parameters(recurse=False)
                       if p.is_floating_point() and not torch.isfinite(p).all()]
            if not in_bad:
                print(f"\n[NAN-DEBUG] ==== FIRST NON-FINITE SOURCE ====", flush=True)
                print(f"[NAN-DEBUG] module   : {name}", flush=True)
                print(f"[NAN-DEBUG] type     : {mod.__class__.__name__}", flush=True)
                print(f"[NAN-DEBUG] inputs OK, OUTPUT non-finite -> produced here", flush=True)
                print(f"[NAN-DEBUG] bad buffers (BN stats etc.): {buf_bad}", flush=True)
                print(f"[NAN-DEBUG] bad params : {par_bad}", flush=True)
                raise RuntimeError(f"[NAN-DEBUG] non-finite produced by module '{name}' ({mod.__class__.__name__})")
        return hook

    for name, m in model.named_modules():
        if name:  # skip the root module
            m.register_forward_hook(make_hook(name, m))
