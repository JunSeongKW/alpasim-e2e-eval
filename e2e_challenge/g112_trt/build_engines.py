"""Build the TensorRT engines of vits512_g112's image backbone (run by build.sh).

Runs inside the axe-v9 image with the g112 bundle mounted exactly as the 441
evaluation did (/app/navsim, /app/nurec_pf, adapter, checkpoint, config), the
TensorRT site at /trt_site, this directory at /g112_trt, trt_bench at /checks,
and an output directory at /out. One GPU.

1. Load the policy as a driver does and run one probe forward (the runtime's own
   probe features) to find the image backbone that actually fires and capture
   its inputs and outputs.
2. Export that module to ONNX as plain fp32 ops (trt_bench/trt_backbone.py).
3. Build two FP16 engines: `exact` for this GPU's compute capability and
   `ampere_plus` (TensorRT hardware-compatibility mode, any cc >= 8.0).
4. Check each against the PyTorch backbone on the probe input and time it.
5. Write manifest.json, which the runtime reads.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/checks")
sys.path.insert(0, "/g112_trt")
from bench_trt import build_policy  # noqa: E402
from trt_backbone import _exportable  # noqa: E402
import trt_backbone_runtime as rt  # noqa: E402

OUT = Path("/out")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": round(sum(times) / len(times), 2), "p50_ms": round(times[len(times) // 2], 2),
            "p99_ms": round(times[int(len(times) * 0.99) - 1], 2)}


def main():
    torch.set_num_threads(1)
    trt = rt._import_tensorrt()
    assert trt is not None, "tensorrt not importable in the build container"
    policy = build_policy()
    agent = policy._agent

    # The backbone that actually runs at inference (teacher and student both have one).
    fired = set()
    hooks = [m.register_forward_hook(lambda mod, a, o, n=n: fired.add(n))
             for n, m in agent.named_modules() if n.split(".")[-1] == "image_backbone"]
    try:
        policy._run_agent(rt._probe_features(policy))
    finally:
        for h in hooks:
            h.remove()
    assert len(fired) == 1, f"expected one image_backbone to fire, got {sorted(fired)}"
    target = fired.pop()
    module = agent.get_submodule(target)
    args, out = rt._capture(policy, module)
    tensors = [a for a in args if torch.is_tensor(a)]
    assert len(tensors) == len(args), "backbone takes non-tensor arguments"
    outs, kind = rt._flatten(out)
    in_names = [f"in{i}" for i in range(len(tensors))]
    out_names = [f"out{i}" for i in range(len(outs))]

    onnx_path = OUT / "backbone.onnx"
    with torch.no_grad(), _exportable(module):
        torch.onnx.export(module.eval(), tuple(t.clone() for t in tensors), str(onnx_path),
                          input_names=in_names, output_names=out_names, opset_version=17,
                          do_constant_folding=True)

    cc = list(torch.cuda.get_device_capability())
    gpu = torch.cuda.get_device_name()
    logger = trt.Logger(trt.Logger.WARNING)
    engines = []
    for eid, compat in (("exact", None), ("ampere_plus", trt.HardwareCompatibilityLevel.AMPERE_PLUS)):
        builder = trt.Builder(logger)
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(net, logger)
        assert parser.parse(onnx_path.read_bytes()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        cfg.set_flag(trt.BuilderFlag.FP16)
        if compat is not None:
            cfg.hardware_compatibility_level = compat
        t0 = time.time()
        serialized = builder.build_serialized_network(net, cfg)
        assert serialized is not None, f"engine build failed: {eid}"
        plan = OUT / f"backbone_{eid}_fp16.plan"
        plan.write_bytes(bytes(serialized))
        engines.append({"id": eid, "kind": eid, "file": plan.name, "sha256": sha256(plan),
                        "bytes": plan.stat().st_size, "build_seconds": round(time.time() - t0, 1),
                        "compute_capability": cc if compat is None else None, "built_on": gpu})

    manifest = {
        "model": "vits512_g112_kv28", "target": target,
        "tensorrt_version": trt.__version__, "torch_version": torch.__version__,
        "inputs": {"names": in_names, "shapes": [list(t.shape) for t in tensors], "dtypes": [str(t.dtype) for t in tensors]},
        "outputs": {"names": out_names, "shapes": [list(o.shape) for o in outs], "dtypes": [str(o.dtype) for o in outs],
                    "kind": kind},
        "onnx_sha256": sha256(onnx_path), "engines": engines,
    }
    runtime = trt.Runtime(logger)
    with torch.inference_mode():
        manifest["timing_torch"] = timed(lambda: module(*args))
        for e in engines:
            engine = runtime.deserialize_cuda_engine((OUT / e["file"]).read_bytes())
            b = rt._TrtBackbone(engine, manifest)
            got, _ = rt._flatten(b(*args))
            e["rel_err_on_build_gpu"] = [round(rt._rel_err(g, r), 5) for g, r in zip(got, outs)]
            e["timing"] = timed(lambda: b(*args))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("target", "tensorrt_version", "inputs", "outputs", "timing_torch")}, indent=1))
    for e in engines:
        print(e["id"], e["bytes"], e["rel_err_on_build_gpu"], e["timing"], e["build_seconds"], "s")


if __name__ == "__main__":
    main()
