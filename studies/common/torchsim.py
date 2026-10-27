"""
PyTorch (GPU) version of the exact anneal, for large banks (S1: 40,000 rows up to p = 12).

It is a line-by-line port of aqfm_prescreen.aqfm.evolve_batch_strang and
observables_batch, so the physics and the conventions are identical:

    H(s) = (1 - s)(-sum X_i) + s * H_data,   H_data diagonal (fields + couplings),
    s = (k + 1/2)/N at step k,  dt = tau / N,  N = strang_steps(tau),
    one step = exp(-i dt/2 s H_data) . prod_q exp(+i dt (1-s) X_q) . exp(-i dt/2 s H_data)
    start |+>^p;  read out X(p), Y(p), Z(p), XX(q), YY(q), ZZ(q)

tests/test_studies_torchsim.py checks it against the NumPy simulator to ~1e-10 (on CPU, and on
GPU when one is present) before anything relies on it.

Precision: complex128 by default (same as NumPy).  The output is cast to float32, exactly as
compute_bank does, so banks from either simulator are interchangeable.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import torch

from aqfm_prescreen.aqfm import data_diagonal_batch, strang_steps, z_signs


def evolve_strang_torch(Z, J, tau, n_steps, device, dtype=torch.complex128):
    """final states (m, 2^p) for every row of Z (m, p); J scalar or (p, p) matrix"""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    m, n = Z.shape
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    # the diagonal of H_data for every row and basis state, computed by the NumPy code itself
    dd = torch.as_tensor(data_diagonal_batch(Z, J, z_signs(n)), dtype=real, device=device)
    dim = 1 << n
    psi = torch.full((m, dim), 1.0 / math.sqrt(dim), dtype=dtype, device=device)    # |+>^n
    one = torch.ones_like(dd)
    dt = float(tau) / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        half = torch.polar(one, (-0.5 * dt * s) * dd)          # exp(-i dt/2 s H_data), elementwise
        psi = psi * half
        th = dt * (1.0 - s)
        c, sn = math.cos(th), 1j * math.sin(th)                # exp(+i th X_q) = cos th + i sin th X_q
        for q in range(n):
            # basis index b viewed as (high bits, bit q, low bits): bit q is axis 2
            a = psi.view(m, 1 << (n - 1 - q), 2, 1 << q)
            psi = (c * a + sn * a.flip(2)).reshape(m, dim)
        psi = psi * half
    return psi


def observables_torch(psi):
    """[X(p), Y(p), Z(p), XX(q), YY(q), ZZ(q)] per row, same conventions as observables_batch:
       X_q = Re <psi|psi[flip q]>,  Y_q = Re <psi| i(-1)^{b_q} psi[flip q]>,
       ZZ from probabilities,  XX / YY via double flips (YY with sign -(-1)^{b_i}(-1)^{b_j})."""
    m, dim = psi.shape
    n = dim.bit_length() - 1
    dev = psi.device
    idx = torch.arange(dim, device=dev)
    ZS = torch.as_tensor(z_signs(n), dtype=torch.float64, device=dev)
    prob = (psi.abs() ** 2).to(torch.float64)
    conj = psi.conj()
    cols = {"X": [], "Y": [], "Z": [], "XX": [], "YY": [], "ZZ": []}
    for q in range(n):
        fl = idx ^ (1 << q)
        sign = (1.0 - 2.0 * ((idx >> q) & 1)).to(torch.float64)
        pf = psi[:, fl]
        cols["X"].append((conj * pf).sum(1).real)
        cols["Y"].append((conj * (1j * sign) * pf).sum(1).real)
    cols["Z"] = [prob @ ZS]
    for i, j in combinations(range(n), 2):
        fl = idx ^ (1 << i) ^ (1 << j)
        si = (1.0 - 2.0 * ((idx >> i) & 1)).to(torch.float64)
        sj = (1.0 - 2.0 * ((idx >> j) & 1)).to(torch.float64)
        pf = psi[:, fl]
        cols["XX"].append((conj * pf).sum(1).real)
        cols["YY"].append((conj * (-(si * sj)) * pf).sum(1).real)
        cols["ZZ"].append(prob @ (ZS[:, i] * ZS[:, j]))
    parts = []
    for k in ("X", "Y", "Z", "XX", "YY", "ZZ"):
        v = cols[k]
        if not v:
            parts.append(torch.empty((m, 0), dtype=torch.float64, device=dev))
        elif k == "Z":
            parts.append(v[0].to(torch.float64))
        else:
            parts.append(torch.stack(v, 1).to(torch.float64))
    return torch.cat(parts, 1)


def default_chunk(p: int) -> int:
    """rows per evolution chunk: ~16.7M amplitudes (~270 MB per complex128 array)"""
    return max(1, (1 << 24) >> p)


def compute_bank_torch(Z, J, tau, device="cuda", n_steps=None, chunk=None, dtype=torch.complex128):
    """Same output as aqfm_prescreen.aqfm.compute_bank(Z, J, tau, method='strang'): float32 bank"""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    m, p = Z.shape
    n_steps = strang_steps(tau) if n_steps is None else n_steps
    chunk = chunk or default_chunk(p)
    out = []
    with torch.no_grad():
        for s in range(0, m, chunk):
            psi = evolve_strang_torch(Z[s:s + chunk], J, tau, n_steps, device, dtype)
            out.append(observables_torch(psi).cpu().numpy())
            del psi
    return np.vstack(out).astype(np.float32)
