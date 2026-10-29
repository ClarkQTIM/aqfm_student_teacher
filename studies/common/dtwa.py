"""
Discrete truncated Wigner approximation (DTWA) for the AQFM anneal: a classical "twin" that
keeps the correlations mean-field drops.  S5 (studies/s5_twin/PLAN.md).

Method: Schachenmayer, Pikovski & Rey, Phys. Rev. X 5, 011022 (2015), arXiv:1408.4441.

The anneal (same conventions as aqfm_prescreen.aqfm):
    H(s) = (1 - s)(-sum_i X_i) + s(-sum_i x_i Z_i - J sum_{i<j} Z_i Z_j),   s = t / tau,
    start |+>^p, Strang steps s = (k + 1/2)/N, dt = tau/N, N = strang_steps(tau).

Each qubit is a classical vector sigma = (sx, sy, sz) in standard Pauli components.  Under
H = -B . sigma it precesses as d(sigma)/dt = 2 sigma x B (the Heisenberg equation with operators
replaced by numbers), with
    B_i = ((1 - s), 0, s h_i),   h_i = x_i + J sum_{j != i} sz_j     (THIS trajectory's sz).
One step = rotate about z by dt s h (field from the current sz; a z-rotation doesn't change sz),
rotate about x by 2 dt (1 - s), rotate about z again with the refreshed field.  That is exactly
mean-field's splitting (aqfm.meanfield_local), with vectors in place of amplitudes.

Mean-field is ONE trajectory started from the unit vector (1, 0, 0).  DTWA averages many
trajectories started from the discrete Wigner points of |+>: sx = +1 and sy, sz = +-1 at random
(each qubit independently), vectors of length sqrt(3).  <P_i> is the trajectory average of P_i,
and <P_i P_j> (i != j) the average of the product.

Readout convention: the bank's Y is -<Y>_standard, exactly as observables_batch and
meanfield_local return it.  tests/test_studies_s5.py checks both limits to 1e-9:
  * one unit-vector trajectory == meanfield_local
  * J = 0, all starting points enumerated == single_qubit_strang (DTWA is exact without coupling)
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np


def initial_points(p: int, M: int, rng=None):
    """(M, p, 3) starting vectors for |+>^p.  If 4^p <= M, every one of the 4^p discrete points
    exactly once (then M is reduced to 4^p); otherwise M independent random draws."""
    if 4 ** p <= M:
        yz = np.array(list(product((1.0, -1.0), repeat=2 * p))).reshape(-1, p, 2)
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        yz = rng.choice(np.array([1.0, -1.0]), size=(M, p, 2))
    S = np.empty((len(yz), p, 3))
    S[..., 0] = 1.0
    S[..., 1:] = yz
    return S


def evolve(X, S0, J, tau, n_steps):
    """classical trajectories.  X (m, p) inputs; S0 (M, p, 3) starting vectors, shared by every
    row.  Returns S (m, M, p, 3) at the end of the anneal (standard components)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, p = X.shape
    S = np.broadcast_to(S0, (m,) + S0.shape).copy()               # (m, M, p, 3)
    sx, sy, sz = S[..., 0], S[..., 1], S[..., 2]
    x = X[:, None, :]                                             # (m, 1, p)
    dt = float(tau) / n_steps

    def field(sz):
        return x + J * (sz.sum(-1, keepdims=True) - sz)           # (m, M, p)

    def zrot(sx, sy, phi):
        c, s = np.cos(phi), np.sin(phi)
        return sx * c + sy * s, sy * c - sx * s

    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        sx, sy = zrot(sx, sy, dt * s * field(sz))
        th = 2.0 * dt * (1.0 - s)
        c, sn = np.cos(th), np.sin(th)
        sy, sz = sy * c + sz * sn, sz * c - sy * sn
        sx, sy = zrot(sx, sy, dt * s * field(sz))
    return np.stack([sx, sy, sz], -1)


def bank_from_trajectories(S, n_use=None):
    """bank (m, 3p + 3q) in compute_bank's layout [X, Y, Z, XX, YY, ZZ] from trajectories
    S (m, M, p, 3); averages over the first n_use trajectories (all if None).  Y follows the
    bank convention (-<Y>_standard); YY is unaffected by that sign."""
    S = S[:, :n_use] if n_use else S
    m, M, p, _ = S.shape
    loc = S.mean(1)                                               # (m, p, 3)
    pairs = list(combinations(range(p), 2))
    ii = np.array([a for a, _ in pairs], int)
    jj = np.array([b for _, b in pairs], int)
    parts = [loc[..., 0], -loc[..., 1], loc[..., 2]]
    for a in range(3):
        C = np.einsum("rmi,rmj->rij", S[..., a], S[..., a]) / M  # (m, p, p) second moments
        parts.append(C[:, ii, jj] if pairs else np.empty((m, 0)))
    return np.hstack(parts)


def dtwa_bank(X, J, tau, n_steps, M, rng=None, chunk=50, n_small=None):
    """DTWA banks for every row of X, in chunks of rows (memory: chunk x M x p x 3 floats).
    Returns (bank over all M trajectories, bank over the first n_small (the full bank when every
    starting point is enumerated), M actually used)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S0 = initial_points(X.shape[1], M, rng)
    if len(S0) == 4 ** X.shape[1]:
        n_small = None          # every starting point enumerated: no sampling noise to check
    out, small = [], []
    for a in range(0, len(X), chunk):
        S = evolve(X[a:a + chunk], S0, J, tau, n_steps)
        out.append(bank_from_trajectories(S))
        if n_small:
            small.append(bank_from_trajectories(S, n_small))
    B = np.vstack(out).astype(np.float32)
    return B, (np.vstack(small).astype(np.float32) if n_small else B), len(S0)
