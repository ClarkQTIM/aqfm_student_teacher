"""
Array backend for the exact statevector simulation.

GPU use is OPT-IN:  export AQFM_GPU=1  (requires CuPy, e.g. pip install cupy-cuda12x).
Each worker process picks a GPU by pid % n_gpus unless AQFM_GPU_DEVICE is set.
Everything else (J=0 factorisation, mean-field, all learning) stays on the CPU.
The math is identical on both backends (complex128 throughout); results agree
to floating-point round-off (verify with experiments/check_gpu.py).
"""
from __future__ import annotations

import os

import numpy as np

_XP = None


def gpu_requested() -> bool:
    return os.environ.get("AQFM_GPU", "0").lower() in ("1", "true", "yes", "on")


def get_xp():
    """numpy, or cupy when AQFM_GPU=1 (raises if requested but unavailable)."""
    global _XP
    if _XP is not None:
        return _XP
    if not gpu_requested():
        _XP = np
        return _XP
    import cupy as cp                                   # noqa: raise if missing
    n = cp.cuda.runtime.getDeviceCount()
    if n == 0:
        raise RuntimeError("AQFM_GPU=1 but no CUDA device is visible")
    dev = int(os.environ.get("AQFM_GPU_DEVICE", os.getpid() % n))
    cp.cuda.Device(dev).use()
    _XP = cp
    return _XP


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def default_chunk(p: int) -> int:
    """Rows per evolution chunk: ~6e6 amplitudes on CPU (~100 MB working set),
    ~6e7 on GPU (~1 GB per array; a few GB peak per worker)."""
    budget = 6e7 if get_xp() is not np else 6e6
    return max(1, int(budget // (1 << p)))
