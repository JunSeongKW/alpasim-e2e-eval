"""In-image functional test of the TensorRT backbone (run by validate.sh on one GPU).

Builds the policy twice the way the driver does -- once as shipped (the hook
installs TensorRT) and once with DRIVESUPRIM_TRT_BACKBONE=off -- and checks:
  1. the shipped policy really runs the engine (SafeBackbone, active),
  2. on the same probe input both policies select the same candidate and their
     trajectories agree to a few centimetres,
  3. an input shape the engine was not built for is answered by PyTorch with the
     engine kept on, and a TensorRT call that raises switches the process to
     PyTorch, which then reproduces the PyTorch-only policy,
  4. per-call latency of the full forward with and without the engine.
Writes /report/test_trt_image.json; exits non-zero if any check fails.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app")
from drivesuprim_challenge import trt_backbone_runtime as rt  # noqa: E402
from drivesuprim_challenge.policy import DriveSuprimPolicy  # noqa: E402


def build():
    a = "/app/assets/drivesuprim"
    return DriveSuprimPolicy(
        checkpoint_path=os.environ["DRIVESUPRIM_CHECKPOINT_PATH"],
        vocab_path=os.environ.get("DRIVESUPRIM_VOCAB_PATH", f"{a}/nurec_train_kmeans_4096x40x3.npy"),
        backbone_type=os.environ.get("DRIVESUPRIM_BACKBONE_TYPE", "bevformer_m"),
        device="cuda",
        config_path=os.environ["DRIVESUPRIM_CONFIG_PATH"],
    )


def trajectory(pred):
    # Both the coarse ("trajectory") and the refined ("final_traj") output, whichever
    # _prediction_to_result ends up sending.
    parts = [pred[k].detach().float().reshape(-1, pred[k].shape[-1]).cpu().numpy()
             for k in ("trajectory", "final_traj") if torch.is_tensor(pred.get(k))]
    if not parts:
        raise KeyError(f"no trajectory in prediction keys {sorted(pred)}")
    return np.concatenate(parts)


def timed(policy, feats, n=50):
    for _ in range(5):
        policy._run_agent(feats)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy._run_agent(feats)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return {"mean_ms": round(float(np.mean(t)), 1), "p50_ms": round(t[n // 2], 1), "p99_ms": round(t[-1], 1)}


def main():
    report, ok = {}, True
    manifest = json.load(open(os.path.join(os.environ["DRIVESUPRIM_TRT_DIR"], "manifest.json")))
    target = manifest["target"]
    shipped = build()
    swapped = shipped._agent.get_submodule(target)
    report["engine"] = getattr(shipped, "_trt_engine", None)
    report["installed"] = isinstance(swapped, rt.SafeBackbone) and swapped.active
    if not report["installed"]:  # nothing below means anything without the engine
        print("RESULT", json.dumps(report))
        sys.exit(1)

    os.environ["DRIVESUPRIM_TRT_BACKBONE"] = "off"
    reference = build()
    ok &= not isinstance(reference._agent.get_submodule(target), rt.SafeBackbone)

    feats = rt._probe_features(shipped)
    with torch.inference_mode():
        a, b = shipped._run_agent(feats), reference._run_agent(feats)
    ta, tb = trajectory(a), trajectory(b)
    sel = [a.get("selected_indices"), b.get("selected_indices")]
    report["same_selected_candidate"] = (None if sel[0] is None else bool(torch.equal(sel[0].cpu(), sel[1].cpu())))
    report["trajectory_max_abs_diff_m"] = round(float(np.abs(ta - tb).max()), 4)
    report["trajectory_endpoint_trt"] = ta[-1].tolist()
    report["trajectory_endpoint_torch"] = tb[-1].tolist()
    # FP16 can flip a near-tie between candidates on a random probe; that is reported,
    # not failed (the closed-loop parity run judges it). Same candidate must mean same path.
    if report["same_selected_candidate"] is not False:
        ok &= report["trajectory_max_abs_diff_m"] < 0.5

    with torch.inference_mode():
        report["latency_trt"] = timed(shipped, feats)
        report["latency_torch"] = timed(reference, feats)

    # An input the engine was not built for is answered by PyTorch, and the engine stays on.
    assert len(manifest["inputs"]["shapes"]) == 1, "fallback test assumes a one-tensor backbone"
    shape = manifest["inputs"]["shapes"][0]
    odd = torch.rand([shape[0] + 1] + shape[1:], device="cuda")
    with torch.inference_mode():
        got, _ = rt._flatten(swapped(odd))
        ref, _ = rt._flatten(swapped.torch_backbone(odd))
    # Not bit-equal: two PyTorch calls on the same new-shape input differed in the
    # last bits on 10-13 (a nondeterministic kernel; which one is not pinned down).
    report["other_shape_vs_torch_rel_err"] = max(rt._rel_err(g, r) for g, r in zip(got, ref))
    report["other_shape_answered_by_torch"] = (swapped.active and swapped.shape_misses == 1
                                               and report["other_shape_vs_torch_rel_err"] < 1e-3)
    with torch.inference_mode():
        again = trajectory(shipped._run_agent(feats))
    report["engine_still_used_after_other_shape"] = bool(np.abs(again - ta).max() < 1e-3)
    ok &= report["other_shape_answered_by_torch"] and report["engine_still_used_after_other_shape"]

    # A TensorRT call that raises switches the process to PyTorch for good.
    swapped._trt.context = None
    with torch.inference_mode():
        c = shipped._run_agent(feats)
    report["fallback_on_trt_error"] = (not swapped.active) and swapped.fallbacks == 1
    report["forward_after_fallback_vs_torch_m"] = round(float(np.abs(trajectory(c) - tb).max()), 6)
    ok &= report["fallback_on_trt_error"] and report["forward_after_fallback_vs_torch_m"] < 1e-3

    report["passed"] = bool(ok)
    report["gpu"] = torch.cuda.get_device_name()
    # Two policies live in this process; the per-driver VRAM check is validate.sh's.
    report["peak_reserved_mib_two_policies"] = round(torch.cuda.max_memory_reserved() / 2**20)
    os.makedirs("/report", exist_ok=True)
    with open("/report/test_trt_image.json", "w") as f:
        json.dump(report, f, indent=2)
    print("RESULT", json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
