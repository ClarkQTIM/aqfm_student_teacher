"""VERBATIM copy of evolution + observables from 13_synthetic_aqfm_frozen_runner.py.
Used only as a reference in tests. Do not edit."""
import numpy as np
from itertools import combinations

def z_signs(n):
    """Return shape (2^n,n), entries +/-1 for computational basis Z eigenvalues."""
    dim = 1 << n
    idx = np.arange(dim, dtype=np.uint64)[:, None]
    bits = (idx >> np.arange(n, dtype=np.uint64)[None, :]) & 1
    return (1.0 - 2.0 * bits).astype(np.float64)


def data_diagonal(x, J, ZS):
    # -sum x_i z_i - J sum_{i<j} z_i z_j
    linear = -(ZS @ x)
    if J == 0:
        return linear
    # sum_{i<j} z_i z_j = ((sum_i z_i)^2 - n)/2
    sz = ZS.sum(axis=1)
    pair = 0.5 * (sz*sz - ZS.shape[1])
    return linear - J * pair


def apply_hinit(v, n):
    """(-sum_i X_i) v."""
    dim = len(v)
    idx = np.arange(dim)
    out = np.zeros_like(v)
    for q in range(n):
        out -= v[idx ^ (1 << q)]
    return out


def rk4_step(psi, dt, s, diag_data, n):
    """
    Fixed midpoint-Hamiltonian propagation using RK4 on dpsi/dt=-i H(s) psi.

    NOTE:
    This is a 400-step midpoint-in-time integration of the time-dependent
    Schrödinger equation. It avoids constructing matrices and is stable at
    these small dt values. The midpoint s is held fixed over the RK4 substep.
    """
    def f(v):
        Hv = (1.0-s) * apply_hinit(v, n) + s * diag_data * v
        return -1j * Hv
    k1 = f(psi)
    k2 = f(psi + 0.5*dt*k1)
    k3 = f(psi + 0.5*dt*k2)
    k4 = f(psi + dt*k3)
    out = psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # Tiny numerical drift only; normalize.
    out /= np.linalg.norm(out)
    return out


def evolve_one(x, J, tau, n_steps, ZS):
    n = len(x)
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    dd = data_diagonal(x, J, ZS)
    dt = tau / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        psi = rk4_step(psi, dt, s, dd, n)
    return psi


# ============================================================================
# PAULI EXPECTATIONS
# ============================================================================

def exp_z(psi, ZS):
    prob = np.abs(psi)**2
    return prob @ ZS


def exp_single_flip(psi, q, phase=None):
    idx = np.arange(len(psi))
    flip = idx ^ (1 << q)
    if phase is None:  # X
        return float(np.vdot(psi, psi[flip]).real)
    # Y |b> = i*(-1)^b |b xor 1> under this indexing convention
    bit = ((idx >> q) & 1)
    fac = 1j * (1.0 - 2.0*bit)
    return float(np.vdot(psi, fac * psi[flip]).real)


def exp_pair_same(psi, i, j, kind, ZS):
    idx = np.arange(len(psi))
    if kind == "ZZ":
        return float(np.sum(np.abs(psi)**2 * ZS[:, i] * ZS[:, j]).real)
    flip = idx ^ (1 << i) ^ (1 << j)
    if kind == "XX":
        return float(np.vdot(psi, psi[flip]).real)
    if kind == "YY":
        bi = ((idx >> i) & 1)
        bj = ((idx >> j) & 1)
        # product of the two Y phase factors
        fac = -(1.0 - 2.0*bi) * (1.0 - 2.0*bj)
        return float(np.vdot(psi, fac * psi[flip]).real)
    raise ValueError(kind)


def observables(psi, ZS):
    n = ZS.shape[1]
    X = np.array([exp_single_flip(psi, q) for q in range(n)])
    Y = np.array([exp_single_flip(psi, q, phase="Y") for q in range(n)])
    Z = exp_z(psi, ZS)
    pairs = list(combinations(range(n), 2))
    XX = np.array([exp_pair_same(psi, i, j, "XX", ZS) for i,j in pairs])
    YY = np.array([exp_pair_same(psi, i, j, "YY", ZS) for i,j in pairs])
    ZZ = np.array([exp_pair_same(psi, i, j, "ZZ", ZS) for i,j in pairs])
    return np.concatenate([X, Y, Z, XX, YY, ZZ])


