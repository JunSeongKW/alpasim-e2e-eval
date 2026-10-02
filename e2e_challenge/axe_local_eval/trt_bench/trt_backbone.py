"""Replace a torch submodule with a TensorRT FP16 engine of itself.

The backbone's input signature is not guessed. One real forward runs with a hook
on the module, and whatever it actually receives -- shape, dtype, how many
tensors -- is what gets exported. Its output structure (a tensor, or a tuple or
list of multi-scale feature maps, which image backbones in BEV models usually
return) is captured the same way and reproduced by the wrapper, so the module can
be swapped in without the caller noticing.

Path: torch.onnx.export -> TensorRT builder (FP16) -> execute_async_v3 on torch's
current stream, reading and writing torch tensors in place. Only the `tensorrt`
package is needed; torch-tensorrt is not, which avoids pinning it to this exact
torch build.

Shapes are fixed (batch of one Drive call). The engine is cached under workdir and
reused when the exported ONNX has not changed.
"""
import hashlib
import importlib
import os
import time

import torch


def import_tensorrt():
    """The TensorRT Python module, under whichever name this install gave it.

    The pip meta-package `tensorrt` is what normally provides `import tensorrt`,
    but it installs its components by spawning a nested pip that ignores
    --target, so it cannot be installed into a mounted site directory. The
    component wheels were installed directly instead, and those expose the module
    as `tensorrt_bindings`. Either name is the same library.
    """
    for name in ("tensorrt", "tensorrt_bindings"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "Builder"):
                return mod
        except ImportError:
            continue
    raise ImportError("TensorRT Python module not found as 'tensorrt' or 'tensorrt_bindings'")


class _Capture:
    def __init__(self):
        self.args = None
        self.out = None

    def __call__(self, module, args, out):
        if self.args is None:
            self.args = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args)
            self.out = out


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


class TrtModule(torch.nn.Module):
    def __init__(self, engine, out_kind, out_shapes, out_dtypes, in_names, out_names):
        super().__init__()
        self.engine = engine
        self.context = engine.create_execution_context()
        self.out_kind = out_kind
        self.out_shapes = out_shapes
        self.out_dtypes = out_dtypes
        self.in_names = in_names
        self.out_names = out_names
        self.build_seconds = 0.0

    @classmethod
    def build(cls, module, *, sample, policy, workdir="/tmp/trt"):
        trt = import_tensorrt()

        os.makedirs(workdir, exist_ok=True)
        cap = _Capture()
        h = module.register_forward_hook(cap)
        try:
            policy._use_autocast = False
            with torch.inference_mode():
                policy._run_agent(sample)
        finally:
            h.remove()
        if cap.args is None:
            raise RuntimeError("backbone was never called during a forward")
        tensor_args = [a for a in cap.args if torch.is_tensor(a)]
        if len(tensor_args) != len(cap.args):
            raise TypeError("backbone takes non-tensor positional arguments; not exportable as-is")
        outs, kind = _flatten(cap.out)

        in_names = [f"in{i}" for i in range(len(tensor_args))]
        out_names = [f"out{i}" for i in range(len(outs))]
        onnx_path = os.path.join(workdir, "backbone.onnx")

        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(
                module.eval(), tuple(tensor_args), onnx_path,
                input_names=in_names, output_names=out_names,
                opset_version=17, do_constant_folding=True,
            )
        digest = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()[:16]
        plan_path = os.path.join(workdir, f"backbone_fp16_{digest}.plan")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        if os.path.exists(plan_path):
            engine = runtime.deserialize_cuda_engine(open(plan_path, "rb").read())
        else:
            builder = trt.Builder(logger)
            net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse(open(onnx_path, "rb").read()):
                errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed: {errs}")
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            cfg.set_flag(trt.BuilderFlag.FP16)
            serialized = builder.build_serialized_network(net, cfg)
            if serialized is None:
                raise RuntimeError("TensorRT engine build failed")
            open(plan_path, "wb").write(bytes(serialized))
            engine = runtime.deserialize_cuda_engine(serialized)

        mod = cls(engine, kind, [o.shape for o in outs], [o.dtype for o in outs], in_names, out_names)
        mod.build_seconds = round(time.time() - t0, 1)
        return mod

    def forward(self, *args):
        stream = torch.cuda.current_stream()
        for name, a in zip(self.in_names, args):
            a = a.contiguous().float()
            self.context.set_input_shape(name, tuple(a.shape))
            self.context.set_tensor_address(name, a.data_ptr())
        outs = []
        for name, shape, dtype in zip(self.out_names, self.out_shapes, self.out_dtypes):
            o = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, o.data_ptr())
            outs.append(o)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute failed")
        outs = [o.to(d) for o, d in zip(outs, self.out_dtypes)]
        return _restore(outs, self.out_kind)
