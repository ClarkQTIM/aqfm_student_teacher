"""
PyTorch (GPU) DTWA for S5 Amendment 2: the same integrator as studies/common/dtwa.py, line for line,
so many more trajectories are affordable.  tests/test_studies_torchdtwa.py checks it against the NumPy
version (which is itself tested against three exact limits) before it is used.

nested_banks() averages over NESTED prefixes of one set of trajectories (M_1 < M_2 < ... <= M), so the
banks at different M differ only by sampling, and the starting points are a random permutation of
the full discrete set when 4^p <= M (every prefix is then a random subset without replacement).
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import torch


def starting_points(p: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """(M_used, p, 3): sx = +1, (sy, sz) = +-1.  If 4^p <= M, all 4^p points in random order
    (M_used = 4^p); otherwise M independent draws."""
    if 4 ** p <= M:
        yz = np.array(list(product((1.0, -1.0), repeat=2 * p))).reshape(-1, p, 2)
        yz = yz[rng.permutation(len(yz))]
    else:
        yz = rng.choice(np.array([1.0, -1.0]), size=(M, p, 2))
    S = np.empty((len(yz), p, 3))
    S[..., 0] = 1.0
    S[..., 1:] = yz
    return S


def evolve_torch(X, S0, J, tau, n_steps, device, dtype=torch.float32):
    """X (m, p), S0 (M, p, 3) -> final trajectories (m, M, p, 3) on `device` (standard components).
    Same steps as dtwa.evolve: z-rotation by dt s h (h from this trajectory's sz), x-rotation by
    2 dt (1 - s), z-rotation with the refreshed field."""
    X = torch.as_tensor(np.atleast_2d(np.asarray(X, dtype=float)), dtype=dtype, device=device)
    S0 = torch.as_tensor(S0, dtype=dtype, device=device)
    m, p = X.shape
    M = S0.shape[0]
    sx = S0[None, :, :, 0].expand(m, M, p).clone()
    sy = S0[None, :, :, 1].expand(m, M, p).clone()
    sz = S0[None, :, :, 2].expand(m, M, p).clone()
    x = X[:, None, :]
    dt = float(tau) / n_steps
    J = float(J)

    def zrot(sx, sy, sz, a):
        phi = a * (x + J * (sz.sum(-1, keepdim=True) - sz))
        c, s = torch.cos(phi), torch.sin(phi)
        return sx * c + sy * s, sy * c - sx * s

    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        sx, sy = zrot(sx, sy, sz, dt * s)
        th = 2.0 * dt * (1.0 - s)
        c, sn = float(np.cos(th)), float(np.sin(th))
        sy, sz = sy * c + sz * sn, sz * c - sy * sn
        sx, sy = zrot(sx, sy, sz, dt * s)
    return torch.stack([sx, sy, sz], -1)


def bank_prefix(S, n):
    """bank (m, 3p + 3q) [X, Y, Z, XX, YY, ZZ] from the first n trajectories of S (m, M, p, 3) (a
    torch tensor); Y in the bank convention.  Returned as float64 numpy."""
    S = S[:, :n].double()
    m, _, p, _ = S.shape
    loc = S.mean(1)
    pairs = list(combinations(range(p), 2))
    ii = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=S.device)
    jj = torch.tensor([b for _, b in pairs], dtype=torch.long, device=S.device)
    parts = [loc[..., 0], -loc[..., 1], loc[..., 2]]
    for a in range(3):
        C = torch.einsum("rmi,rmj->rij", S[..., a], S[..., a]) / n
        parts.append(C[:, ii, jj] if pairs else torch.zeros(m, 0, dtype=S.dtype, device=S.device))
    return torch.cat(parts, 1).cpu().numpy()


def nested_banks(X, J, tau, n_steps, Ms, rng, device, chunk=50, dtype=torch.float32):
    """{M_k: bank (m, 3p + 3q)} for nested prefixes Ms (ascending; capped at the number of starting
    points), plus 'loc_var': per-row trajectory variance of the 3p single-qubit values at the largest
    M (for the signal-to-noise diagnostic).  Rows go through in chunks."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = X.shape[1]
    S0 = starting_points(p, max(Ms), rng)
    Ms = sorted({min(M, len(S0)) for M in Ms})
    out = {M: [] for M in Ms}
    var = []
    for a in range(0, len(X), chunk):
        S = evolve_torch(X[a:a + chunk], S0, J, tau, n_steps, device, dtype)
        for M in Ms:
            out[M].append(bank_prefix(S, M))
        v = S.double().var(1)                                       # (m, p, 3) over trajectories
        var.append(torch.cat([v[..., 0], v[..., 1], v[..., 2]], 1).cpu().numpy())
        del S
    res = {M: np.vstack(b) for M, b in out.items()}
    res["loc_var"] = np.vstack(var)
    res["M_used"] = len(S0)
    return res
