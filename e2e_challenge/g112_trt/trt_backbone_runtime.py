"""TensorRT FP16 engine for the ViT image backbone, PyTorch always kept as the safe path.

install(policy) is called once from DriveSuprimPolicy.__init__, before the driver
reports ready, and never raises. Anything that goes wrong -- the tensorrt module
missing, a TensorRT version other than the engines were built with, a GPU the
engines cannot run on, an engine that fails to deserialize, or outputs that do not
match the PyTorch backbone on this GPU -- leaves the PyTorch backbone in place and
prints why. Once installed, an input the engine was not built for goes to the
PyTorch backbone for that call, and a TensorRT call that raises is answered by
the PyTorch backbone, which the process then keeps for good.

Engines (manifest.json next to them, written by e2e_challenge/g112_trt/build_engines.py):
  exact        built for one compute capability (the build GPU's); used only there
  ampere_plus  TensorRT hardware-compatibility mode; runs on any compute capability
               >= 8.0 (Ampere, Ada, Hopper), a little slower

Environment (baked into the image; the evaluator sets none of these):
  DRIVESUPRIM_TRT_BACKBONE      auto (default) | off
  DRIVESUPRIM_TRT_DIR           directory holding manifest.json and the .plan files
  DRIVESUPRIM_TRT_MAX_REL_ERR   validation threshold on relative L2 error (default 0.05)
  DRIVESUPRIM_TRT_ENGINE        test only: try this engine id and no other
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

_PREFIX = "[DriveSuprim] TRT backbone:"


def _log(message):
    print(f"{_PREFIX} {message}", flush=True)


_IMPORT_ERRORS = []


def _import_tensorrt():
    for name in ("tensorrt", "tensorrt_bindings"):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- a broken install is just "unavailable"
            _IMPORT_ERRORS.append(f"{name}: {exc!r}")
            continue
        if hasattr(mod, "Runtime"):
            return mod
        _IMPORT_ERRORS.append(f"{name}: no Runtime")
    return None


def _flatten(out):
    if torch.is_tensor(out):
        return [out], "tensor"
    if isinstance(out, (list, tuple)) and all(torch.is_tensor(o) for o in out):
        return list(out), type(out).__name__
    raise TypeError(f"unsupported backbone output type {type(out)}")


def _restore(tensors, kind):
    if kind == "tensor":
        return tensors[0]
    return tuple(tensors) if kind == "tuple" else list(tensors)


def _dtype(name):
    return getattr(torch, name.replace("torch.", ""))


class _TrtBackbone:
    """Runs one deserialized engine on torch tensors, on torch's current stream."""

    def __init__(self, engine, spec):
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("could not create an execution context")
        self.in_names = spec["inputs"]["names"]
        self.in_shapes = [tuple(s) for s in spec["inputs"]["shapes"]]
        self.out_names = spec["outputs"]["names"]
        self.out_shapes = [tuple(s) for s in spec["outputs"]["shapes"]]
        self.out_dtypes = [_dtype(d) for d in spec["outputs"]["dtypes"]]
        self.out_kind = spec["outputs"]["kind"]

    def accepts(self, args):
        return (len(args) == len(self.in_names) and all(torch.is_tensor(a) and a.is_cuda for a in args)
                and [tuple(a.shape) for a in args] == self.in_shapes)

    def __call__(self, *args):
        if not self.accepts(args):
            raise RuntimeError(f"inputs {[getattr(a, 'shape', type(a)) for a in args]} != engine {self.in_shapes}")
        tensors = list(args)
        stream = torch.cuda.current_stream()
        keep = []
        for name, a in zip(self.in_names, tensors):
            a = a.contiguous().float()
            keep.append(a)
            self.context.set_input_shape(name, tuple(a.shape))
            self.context.set_tensor_address(name, a.data_ptr())
        outs = []
        for name, shape in zip(self.out_names, self.out_shapes):
            o = torch.empty(shape, dtype=torch.float32, device=tensors[0].device)
            self.context.set_tensor_address(name, o.data_ptr())
            outs.append(o)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")
        return _restore([o.to(d) for o, d in zip(outs, self.out_dtypes)], self.out_kind)


class SafeBackbone(torch.nn.Module):
    """Drop-in for the backbone: TensorRT while it works, the original module otherwise.

    An input the engine was not built for goes to PyTorch for that call only. A
    TensorRT call that raises switches the process to PyTorch for good.
    """

    def __init__(self, trt_backbone, torch_backbone, engine_id):
        super().__init__()
        self.torch_backbone = torch_backbone
        self._trt = trt_backbone
        self.engine_id = engine_id
        self.active = True
        self.calls = 0
        self.shape_misses = 0
        self.fallbacks = 0

    def forward(self, *args):
        self.calls += 1
        if self.active:
            if not self._trt.accepts(args):
                self.shape_misses += 1
                if self.shape_misses <= 3:
                    _log(f"call {self.calls}: input {[getattr(a, 'shape', type(a)) for a in args]} "
                         f"is not the engine's; PyTorch for this call")
            else:
                try:
                    return self._trt(*args)
                except Exception as exc:  # noqa: BLE001 -- any failure falls back
                    self.active = False
                    self.fallbacks += 1
                    _log(f"call {self.calls} failed ({exc!r}); PyTorch backbone for the rest of this process")
        return self.torch_backbone(*args)


def _probe_features(policy):
    """Synthetic inputs shaped exactly like a Drive call (as in pkg3cam_eval/smoke_pkg.py)."""
    cfg = policy._config
    W, H, T = int(cfg.bev_img_width), int(cfg.bev_img_height), int(cfg.bev_seq_len)
    n_cam = int(cfg.bev_num_cameras)
    g = torch.Generator().manual_seed(0)
    imgs = torch.rand(1, T, n_cam, 3, H, W, generator=g)
    sx, sy = W / 512.0, H / 256.0
    K = np.array([[377.0 * sx, 0, 256.0 * sx], [0, 377.0 * sy, 128.0 * sy], [0, 0, 1.0]])
    mats = []
    for yaw in np.linspace(60.0, -60.0, n_cam):
        y = np.radians(yaw)
        R = np.stack([[np.sin(y), -np.cos(y), 0.0], [0.0, 0.0, -1.0], [np.cos(y), np.sin(y), 0.0]])
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = -R @ np.array([1.5, 0.0, 1.6])
        P = np.eye(4)
        P[:3, :3] = K
        mats.append(P @ E)
    cams = np.stack(mats).astype(np.float32)
    s = np.linspace(42.0, 80.0, 20, dtype=np.float32)
    feats = {
        "bev_imgs": imgs,
        "bev_imgs_teacher": imgs.clone(),
        "lidar2img": torch.from_numpy(np.stack([cams] * T))[None],
        "bev_ego_pose": torch.zeros(1, T, 3),
        "status_feature": [torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0]])] * 2,
        "route_feature": torch.from_numpy(np.stack([s, np.zeros_like(s)], -1) / 80.0)[None],
        "route_mask": torch.ones(1, 20),
        "ego_geom": torch.tensor([[4.6 * 0.98, 1.9 * 0.98, 1.467]]),
    }
    dev = policy._device
    return {k: ([t.float().to(dev) for t in v] if isinstance(v, list) else v.float().to(dev)) for k, v in feats.items()}


def _capture(policy, module):
    box = {}

    def hook(mod, args, out):
        if "args" not in box:
            box["args"] = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args)
            box["out"] = out

    handle = module.register_forward_hook(hook)
    try:
        policy._run_agent(_probe_features(policy))
    finally:
        handle.remove()
    if "args" not in box:
        raise RuntimeError("backbone did not run during the probe forward")
    return box["args"], box["out"]


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install(policy):
    """Swap the image backbone for a validated TensorRT engine; never raises."""
    try:
        return _install(policy)
    except Exception as exc:  # noqa: BLE001 -- the PyTorch path must survive anything
        _log(f"not installed ({exc!r}); PyTorch backbone")
        return None
    finally:
        # The probe forwards ran the whole agent. This model keeps no state between
        # calls (bev_streaming off), but drop any BEV stream cache regardless so the
        # first real frame can never see the probe's.
        try:
            for m in policy._agent.modules():
                if callable(getattr(m, "reset_stream", None)):
                    m.reset_stream()
        except Exception:  # noqa: BLE001
            pass


def _install(policy):
    mode = os.environ.get("DRIVESUPRIM_TRT_BACKBONE", "auto").strip().lower()
    if mode == "off":
        _log("disabled (DRIVESUPRIM_TRT_BACKBONE=off); PyTorch backbone")
        return None
    if policy._device.type != "cuda":
        _log("no CUDA device; PyTorch backbone")
        return None
    root = Path(os.environ.get("DRIVESUPRIM_TRT_DIR", "/app/assets/drivesuprim/trt"))
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        _log(f"no manifest at {manifest_path}; PyTorch backbone")
        return None
    manifest = json.loads(manifest_path.read_text())
    trt = _import_tensorrt()
    if trt is None:
        _log(f"tensorrt module not importable ({'; '.join(_IMPORT_ERRORS)}); PyTorch backbone")
        return None
    if trt.__version__ != manifest["tensorrt_version"]:
        _log(f"TensorRT {trt.__version__} != engines' {manifest['tensorrt_version']}; PyTorch backbone")
        return None

    target = manifest["target"]
    parent_name, _, leaf = target.rpartition(".")
    parent = policy._agent.get_submodule(parent_name) if parent_name else policy._agent
    torch_backbone = getattr(parent, leaf)
    args, out = _capture(policy, torch_backbone)
    shapes = [list(a.shape) for a in args if torch.is_tensor(a)]
    if shapes != manifest["inputs"]["shapes"]:
        _log(f"probe input shapes {shapes} != engines' {manifest['inputs']['shapes']}; PyTorch backbone")
        return None
    ref, _ = _flatten(out)

    cc = list(torch.cuda.get_device_capability(policy._device))
    forced = os.environ.get("DRIVESUPRIM_TRT_ENGINE", "").strip()
    order = [e for e in manifest["engines"] if e["kind"] == "exact" and e["compute_capability"] == cc]
    if cc >= [8, 0]:
        order += [e for e in manifest["engines"] if e["kind"] == "ampere_plus"]
    if forced:
        order = [e for e in manifest["engines"] if e["id"] == forced]
    if not order:
        _log(f"no engine for compute capability {cc}; PyTorch backbone")
        return None

    threshold = float(os.environ.get("DRIVESUPRIM_TRT_MAX_REL_ERR", "0.05"))
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    gpu = torch.cuda.get_device_name(policy._device)
    for spec in order:
        plan = root / spec["file"]
        try:
            if _sha256(plan) != spec["sha256"]:
                raise RuntimeError("plan file checksum mismatch")
            engine = runtime.deserialize_cuda_engine(plan.read_bytes())
            if engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")
            candidate = _TrtBackbone(engine, manifest)
            with torch.inference_mode():
                got, _ = _flatten(candidate(*args))
                torch.cuda.synchronize(policy._device)
            errs = [_rel_err(g, r) for g, r in zip(got, ref)]
            if not all(np.isfinite(errs)) or max(errs) > threshold:
                raise RuntimeError(f"validation failed: rel_err {[round(e, 4) for e in errs]} > {threshold}")
        except Exception as exc:  # noqa: BLE001 -- try the next engine
            _log(f"engine {spec['id']} rejected on {gpu} (cc {cc}): {exc!r}")
            continue
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(3):
                candidate(*args)
        torch.cuda.synchronize(policy._device)
        setattr(parent, leaf, SafeBackbone(candidate, torch_backbone, spec["id"]))
        _log(f"active engine={spec['id']} gpu={gpu} cc={cc} trt={trt.__version__} "
             f"rel_err={[round(e, 4) for e in errs]} warm_ms={round((time.time() - t0) / 3 * 1000, 1)}")
        return spec["id"]
    _log(f"every engine rejected on {gpu} (cc {cc}); PyTorch backbone")
    return None
