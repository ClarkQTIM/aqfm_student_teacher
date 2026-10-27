"""
A shared store of simulated feature banks, so no study simulates the same thing twice.

The input pool
    For each number of qubits p there is ONE fixed pool of POOL_ROWS input rows,
    x ~ N(0, I_p) drawn in a single call from a generator seeded by the name
    'studies_pool_p{p}', then z = clip(x, -3, 3).  Nothing is standardized by
    the sample, so a row's value never depends on how many rows are used.
    S2 uses rows 0-399; S1 will use rows 0-39,999 (the first 30,000 for
    training, the last 10,000 as its test set).

Blocks and pieces
    A bank (one simulator `kind`, p, c, tau) is stored in BLOCK-row blocks:
        <root>/p{p}/{kind}_c{c}_t{tau}/block{b:04d}.npy
    Each block is the output of aqfm_prescreen.aqfm.compute_bank on pool rows
    [b*BLOCK, (b+1)*BLOCK): columns X(p), Y(p), Z(p), XX(q), YY(q), ZZ(q).
    Large p is slow (about 50 s per row at p = 16, tau = 20), so a block is
    computed as several PIECES, one parallel job each, then merged:
        .../block{b:04d}.rows{start}-{stop}.npy   (deleted after the merge)
    Every file is written to a temporary name and renamed, so a killed run
    never leaves a half-written file that looks finished.  Reruns skip what exists.

Kinds
    Q   exact (Strang splitting, the tau sweep's step rule)
    MF  mean-field (product state, same step rule)
    J0  no coupling (exact product state)
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from aqfm_prescreen.aqfm import compute_bank
from studies.common.cells import coupling

POOL_ROWS = 40_000
BLOCK = 400
CLIP = 3.0
METHOD = {"Q": "strang", "MF": "meanfield", "J0": "strang"}
# If this is set (e.g. "cuda:0"), exact (Q) banks are simulated with PyTorch on that device
# (studies/common/torchsim.py), one whole block per job.  Mean-field and J = 0 stay on CPU.
TORCH_ENV = "STUDIES_TORCH_DEVICE"


def torch_device():
    return os.environ.get(TORCH_ENV) or None


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def pool_inputs(p: int) -> np.ndarray:
    """the fixed (POOL_ROWS x p) input pool for p qubits, clipped at +-3"""
    x = np.random.default_rng(_seed(f"studies_pool_p{p}")).standard_normal((POOL_ROWS, p))
    return np.clip(x, -CLIP, CLIP)


def piece_rows(p: int, kind: str = "Q") -> int:
    """rows per parallel job, sized so one job takes minutes, not hours.
    Mean-field and J = 0 cost O(p) per row, so they are never split; on GPU neither is Q."""
    if p <= 10 or kind != "Q" or torch_device():
        return BLOCK
    return {11: 200, 12: 100, 13: 50, 14: 25, 15: 16}.get(p, 8)


def bank_dir(root: Path, p: int, kind: str, c: float, tau: float) -> Path:
    cc = 0.0 if kind == "J0" else c
    return Path(root) / f"p{p}" / f"{kind}_c{cc:g}_t{tau:g}"


def block_path(root, p, kind, c, tau, b) -> Path:
    return bank_dir(root, p, kind, c, tau) / f"block{b:04d}.npy"


def piece_specs(root, p, kind, c, tau, b):
    """the (start, stop) row ranges, within the pool, that make up block b"""
    lo, hi = b * BLOCK, (b + 1) * BLOCK
    step = piece_rows(p, kind)
    return [(s, min(s + step, hi)) for s in range(lo, hi, step)]


def _piece_path(root, p, kind, c, tau, b, start, stop) -> Path:
    return bank_dir(root, p, kind, c, tau) / f"block{b:04d}.rows{start}-{stop}.npy"


def _save(f: Path, A: np.ndarray):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp.npy")
    np.save(tmp, A)
    tmp.replace(f)


def compute_piece(args):
    """one parallel job: simulate pool rows [start, stop) for one bank"""
    root, p, kind, c, tau, b, start, stop = args
    if block_path(root, p, kind, c, tau, b).exists():
        return "skip", f"p{p} {kind} c{c:g} t{tau:g} rows {start}-{stop}"
    f = _piece_path(root, p, kind, c, tau, b, start, stop)
    if f.exists():
        return "skip", f.name
    t0 = time.time()
    Z = pool_inputs(p)[start:stop]
    dev = torch_device()
    if dev and kind == "Q":
        from studies.common.torchsim import compute_bank_torch      # imported only on GPU workers
        B = compute_bank_torch(Z, coupling(p, c, kind), tau, device=dev)
    else:
        B = compute_bank(Z, coupling(p, c, kind), tau, method=METHOD[kind])
    _save(f, np.asarray(B, dtype=np.float32))
    return f"ok {time.time() - t0:.0f}s", f"p{p} {kind} c{c:g} t{tau:g} rows {start}-{stop}"


def merge_block(root, p, kind, c, tau, b) -> bool:
    """join a block's pieces into block{b}.npy once all exist; True if the block is complete"""
    out = block_path(root, p, kind, c, tau, b)
    if out.exists():
        return True
    specs = piece_specs(root, p, kind, c, tau, b)
    files = [_piece_path(root, p, kind, c, tau, b, s, e) for s, e in specs]
    if not all(f.exists() for f in files):
        return False
    A = np.vstack([np.load(f) for f in files])
    assert A.shape[0] == BLOCK, (A.shape, out)
    _save(out, A)
    for f in files:
        f.unlink()
    return True


def load_rows(root, p, kind, c, tau, n_blocks=1) -> np.ndarray:
    """the first n_blocks * BLOCK rows of one bank (raises if a block is missing)"""
    return np.vstack([np.load(block_path(root, p, kind, c, tau, b)) for b in range(n_blocks)])
