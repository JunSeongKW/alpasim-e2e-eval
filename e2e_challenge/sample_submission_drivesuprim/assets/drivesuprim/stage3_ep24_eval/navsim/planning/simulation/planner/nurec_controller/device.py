"""Where the controller's linear algebra runs.

The ADMM iteration is thousands of small dense products. On numpy that runs at
about 1.4 GFLOPS -- per-matrix call overhead dominates at this size -- against
roughly a hundred times that on a GPU, measured on the same batch. Everything
downstream therefore works in torch, falling back to the CPU when no GPU is
visible so the code stays runnable anywhere.
"""

from __future__ import annotations

import os

import torch

def _resolve_dtype() -> torch.dtype:
    """Precision the rollout runs in, from ``NUREC_CONTROLLER_DTYPE``.

    float64 is the default and is what the port was matched against the
    reference in. float32 was expected to stall the ADMM iteration, which
    terminates on a 1e-4 residual, but measured otherwise: on an A6000 it is
    3.1x faster once tokens are batched, and across scored scenes it moves the
    score of a handful of proposals in four thousand, leaving the tied score
    bands the training consumes identical. Set it to "float32" to take that
    trade; anything else is an error rather than a silent fallback.
    """
    requested = os.environ.get("NUREC_CONTROLLER_DTYPE", "float64").strip().lower()
    if requested in ("float64", "double", "fp64"):
        return torch.float64
    if requested in ("float32", "single", "fp32"):
        return torch.float32
    raise ValueError(
        f"NUREC_CONTROLLER_DTYPE must be float64 or float32, got {requested!r}")


DTYPE = _resolve_dtype()


def resolve_device(device=None) -> torch.device:
    """Device to run on, honouring ``NUREC_CONTROLLER_DEVICE`` when set.

    Scoring runs as many worker processes on one node, so ``cuda`` alone would
    pile every one of them onto device zero. Workers spread themselves across
    the visible GPUs instead; set ``NUREC_CONTROLLER_DEVICE`` to pin one.
    """
    if device is not None:
        return torch.device(device)
    requested = os.environ.get("NUREC_CONTROLLER_DEVICE")
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if count <= 1:
        return torch.device("cuda:0")
    # PID % count spreads badly: Ray spawns its workers with a near-constant PID
    # stride, so a whole pool can land on one card and exhaust it while the others
    # sit idle (observed 2026-08-27: 16 workers, all on cuda:0, 139 GiB OOM).
    # Hash the pid instead -- same cost, and it does not resonate with the stride.
    import zlib
    return torch.device(f"cuda:{zlib.crc32(str(os.getpid()).encode()) % count}")


def as_tensor(array, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=DTYPE, device=device)
