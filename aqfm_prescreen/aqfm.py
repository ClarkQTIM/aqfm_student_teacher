"""
AQFM statevector evolution and Pauli read-out.

This is a FAITHFUL PORT of the evolution used in
    13_synthetic_aqfm_frozen_runner.py
    14_real_aqfm_frozen_runner.py

Physics (unchanged):
    H_init        = -sum_i X_i
    H_data(x; J)  = -sum_i x_i Z_i  -  J sum_{i<j} Z_i Z_j
    H(s)          = (1 - s) H_init + s H_data,     s = t / tau
    |psi(0)>      = |+>^{(x) p}
    integrator    = n_steps midpoint-in-time steps; within each step the
                    midpoint Hamiltonian is held fixed and the state is advanced
                    by ONE classical RK4 step, then renormalised.

The only change vs the original scripts is ENGINEERING, not math:
    * `evolve_batch_rk4` advances many samples at once (psi has shape
      (n_samples, 2^p)).  Each row undergoes exactly the arithmetic that
      `evolve_one` applied to a single vector, so results match to float
      round-off.  This makes local validation feasible on a laptop.

An additional, independent integrator `evolve_exact` (piecewise-constant
midpoint Hamiltonian, exact matrix exponential per step) is provided ONLY for
verifying the RK4 integrator.  It is not used to generate study features.

Basis / bit conventions (unchanged):
    computational-basis index b; qubit q is bit q of b (little-endian);
    Z eigenvalue of qubit q = +1 if bit is 0, -1 if bit is 1.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import expm

from .backend import default_chunk, get_xp, to_numpy

# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

def z_signs(n: int) -> np.ndarray:
    """(2^n, n) array of +/-1: Z eigenvalue of every qubit in every basis state."""
    dim = 1 << n
    idx = np.arange(dim, dtype=np.uint64)[:, None]
    bits = (idx >> np.arange(n, dtype=np.uint64)[None, :]) & 1
    return (1.0 - 2.0 * bits).astype(np.float64)


def data_diagonal(x: np.ndarray, J: float, ZS: np.ndarray) -> np.ndarray:
    """Diagonal of H_data for one sample:  -sum x_i z_i - J sum_{i<j} z_i z_j.

    Uses the identity  sum_{i<j} z_i z_j = ((sum_i z_i)^2 - n) / 2,
    i.e. ALL-TO-ALL UNIFORM coupling (not data-dependent J_ij).
    """
    linear = -(ZS @ x)
    if J == 0:
        return linear
    sz = ZS.sum(axis=1)
    pair = 0.5 * (sz * sz - ZS.shape[1])
    return linear - J * pair


def _is_matrix(J) -> bool:
    return isinstance(J, np.ndarray) and J.ndim == 2


def coupling_diagonal(Jm: np.ndarray, ZS: np.ndarray) -> np.ndarray:
    """sum_{i<j} J_ij z_i z_j for every basis state, for a symmetric coupling
    matrix Jm (diagonal ignored).  Uniform Jm = J (1 - I) reproduces the
    scalar formula exactly."""
    Jm = np.asarray(Jm, dtype=float)
    Jm = 0.5 * (Jm + Jm.T)
    Jm = Jm - np.diag(np.diag(Jm))
    return 0.5 * np.einsum("bi,ij,bj->b", ZS, Jm, ZS, optimize=True)


def data_diagonal_batch(X: np.ndarray, J, ZS: np.ndarray) -> np.ndarray:
    """Row-wise `data_diagonal` for a batch X of shape (m, n) -> (m, 2^n).
    J is a scalar (uniform all-to-all) or an (n, n) coupling matrix J_ij."""
    linear = -(X @ ZS.T)
    if _is_matrix(J):
        return linear - coupling_diagonal(J, ZS)[None, :]
    if J == 0:
        return linear
    sz = ZS.sum(axis=1)
    pair = 0.5 * (sz * sz - ZS.shape[1])
    return linear - J * pair[None, :]


def _flip_index(n: int) -> list[np.ndarray]:
    """flip[q][b] = b with bit q toggled (action of X_q on basis labels)."""
    idx = np.arange(1 << n)
    return [idx ^ (1 << q) for q in range(n)]


# ---------------------------------------------------------------------------
# Faithful RK4 integrator (batched)
# ---------------------------------------------------------------------------

def evolve_batch_rk4(X: np.ndarray, J: float, tau: float, n_steps: int = 400
                     ) -> np.ndarray:
    """Evolve every row of X (shape (m, p)) and return psi of shape (m, 2^p).

    Arithmetic per row is identical to the original `evolve_one`:
        f(v)  = -i [ (1-s) * H_init v + s * diag_data * v ]
        RK4 with step dt, s held at the step midpoint, then renormalise.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, n = X.shape
    ZS = z_signs(n)
    flips = _flip_index(n)
    dd = data_diagonal_batch(X, J, ZS)                  # (m, dim)
    psi = np.full((m, 1 << n), 1.0 / np.sqrt(1 << n), dtype=np.complex128)
    dt = float(tau) / n_steps

    def apply_hinit(v):                                 # (-sum_q X_q) v
        out = np.zeros_like(v)
        for f in flips:
            out -= v[:, f]
        return out

    for k in range(n_steps):
        s = (k + 0.5) / n_steps

        def f(v):
            return -1j * ((1.0 - s) * apply_hinit(v) + s * dd * v)

        k1 = f(psi)
        k2 = f(psi + 0.5 * dt * k1)
        k3 = f(psi + 0.5 * dt * k2)
        k4 = f(psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    return psi


# ---------------------------------------------------------------------------
# Unitary split-step integrator (candidate replacement for RK4; NOT the frozen
# protocol until explicitly adopted -- see docs/PROTOCOL_NOTES.md, issue 1)
# ---------------------------------------------------------------------------

def evolve_batch_strang(X: np.ndarray, J: float, tau: float, n_steps: int = 400
                        ) -> np.ndarray:
    """Same midpoint-in-time discretisation as the frozen protocol, but each
    step applies exp(-i dt H(s_mid)) via Strang splitting:

        exp(-i dt/2 * s*H_data) . exp(-i dt * (1-s)*H_init) . exp(-i dt/2 * s*H_data)

    - H_data is diagonal  -> exact phase multiply.
    - H_init = -sum X_q    -> exact product of single-qubit X rotations:
          exp(+i theta X_q) = cos(theta) I + i sin(theta) X_q,  theta = dt(1-s).
    Every factor is exactly unitary, so the scheme is unconditionally stable
    (no silent amplitude corruption at large ||H|| dt, unlike RK4).
    Splitting error is O(dt^3 * commutators) per step.
    """
    xp = get_xp()                                # numpy, or cupy if AQFM_GPU=1
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, n = X.shape
    ZS = z_signs(n)
    dd = xp.asarray(data_diagonal_batch(X, J, ZS))
    psi = xp.full((m, 1 << n), 1.0 / np.sqrt(1 << n), dtype=xp.complex128)
    dt = float(tau) / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        half = xp.exp(-0.5j * dt * s * dd)
        psi *= half
        th = dt * (1.0 - s)
        c, sn = np.cos(th), 1j * np.sin(th)
        for q in range(n):                       # commuting single-qubit rotations
            # view basis index b as (high bits, bit q, low bits); bit q is axis 2
            a = psi.reshape(m, 1 << (n - 1 - q), 2, 1 << q)
            psi = (c * a + sn * a[:, :, ::-1, :]).reshape(m, -1)
        psi *= half
    return psi


def strang_steps(tau: float) -> int:
    """Protocol v2 step rule: dt <= 1/160 (=> 1600 steps at tau=10), min 400."""
    return max(400, int(np.ceil(160.0 * tau)))


def single_qubit_strang(x: np.ndarray, tau: float, n_steps: int) -> np.ndarray:
    """J = 0 exact factorisation: each qubit evolves alone under
    -(1-s) X - s x Z from |+>.  Same Strang scheme as `evolve_batch_strang`
    restricted to one qubit.  Returns (len(x), 3) = (<X>, <Y>, <Z>) using the
    SAME sign conventions as `observables_batch` (note: that convention's
    single-qubit Y is -<Y>_standard; harmless for linear read-out, and YY
    products are unaffected)."""
    x = np.asarray(x, dtype=float).ravel()
    a0 = np.full(x.shape, 1 / np.sqrt(2), dtype=np.complex128)
    a1 = a0.copy()
    dt = float(tau) / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        # diagonal of -s x Z: |0> -> -s x, |1> -> +s x
        h0 = np.exp(-0.5j * dt * (-s * x)); h1 = np.exp(-0.5j * dt * (s * x))
        a0 *= h0; a1 *= h1
        th = dt * (1.0 - s)
        c, sn = np.cos(th), 1j * np.sin(th)
        a0, a1 = c * a0 + sn * a1, c * a1 + sn * a0
        a0 *= h0; a1 *= h1
    w = np.conj(a0) * a1
    Xv = 2 * np.real(w)
    Yv = -2 * np.imag(w)            # matches observables_batch convention
    Zv = np.abs(a0) ** 2 - np.abs(a1) ** 2
    return np.stack([Xv, Yv, Zv], axis=1)


def meanfield_local(X: np.ndarray, J: float, tau: float, n_steps: int) -> np.ndarray:
    """MEAN-FIELD (product-state) version of the anneal: qubit i evolves alone
    under -(1-s) X_i - s h_i(t) Z_i with the self-consistent field
        h_i(t) = x_i + J * sum_{j != i} <Z_j>(t),
    which is the exact effective field of -J sum_{i<j} Z_i Z_j on a product
    state.  Same Strang scheme as `single_qubit_strang` (field refreshed at
    each half-step).  Classical, O(p) per row per step, no entanglement.
    J = 0 reduces exactly to `single_qubit_strang`.  Returns (m, p, 3)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a0 = np.full(X.shape, 1 / np.sqrt(2), dtype=np.complex128)
    a1 = a0.copy()
    dt = float(tau) / n_steps
    if _is_matrix(J):                       # h_i = x_i + sum_{j != i} J_ij <Z_j>
        Jm = 0.5 * (J + J.T); Jm = Jm - np.diag(np.diag(Jm))

        def field():
            z = np.abs(a0) ** 2 - np.abs(a1) ** 2
            return X + z @ Jm
    else:
        def field():
            z = np.abs(a0) ** 2 - np.abs(a1) ** 2
            return X + J * (z.sum(axis=1, keepdims=True) - z)

    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        h = field()
        a0 *= np.exp(0.5j * dt * s * h); a1 *= np.exp(-0.5j * dt * s * h)
        th = dt * (1.0 - s)
        c, sn = np.cos(th), 1j * np.sin(th)
        a0, a1 = c * a0 + sn * a1, c * a1 + sn * a0
        h = field()
        a0 *= np.exp(0.5j * dt * s * h); a1 *= np.exp(-0.5j * dt * s * h)
    w = np.conj(a0) * a1
    return np.stack([2 * np.real(w), -2 * np.imag(w),
                     np.abs(a0) ** 2 - np.abs(a1) ** 2], axis=2)


# ---------------------------------------------------------------------------
# Independent reference integrator (verification only)
# ---------------------------------------------------------------------------

def evolve_exact(x: np.ndarray, J: float, tau: float, n_steps: int = 400
                 ) -> np.ndarray:
    """Piecewise-constant midpoint Hamiltonian, EXACT exp per step (dense).

    Only for small p (verification). With n_steps -> large this converges to
    the true time-ordered evolution, so it also measures RK4 error at 400 steps.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    dim = 1 << n
    ZS = z_signs(n)
    Hinit = np.zeros((dim, dim))
    for f in _flip_index(n):
        Hinit[np.arange(dim), f] -= 1.0
    Hdat = np.diag(data_diagonal(x, J, ZS))
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    dt = float(tau) / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        psi = expm(-1j * dt * ((1 - s) * Hinit + s * Hdat)) @ psi
    return psi


# ---------------------------------------------------------------------------
# Pauli read-out (batched, same conventions as original)
# ---------------------------------------------------------------------------

def observables_batch(psi: np.ndarray) -> np.ndarray:
    """Return [X(p), Y(p), Z(p), XX(q), YY(q), ZZ(q)] for each row of psi.

    Ordering and sign conventions match `observables()` in the originals:
        X_q  = Re <psi| psi[flip_q]>
        Y_q  = Re <psi| i(-1)^{b_q} psi[flip_q]>
        ZZ   from probabilities, XX/YY via double flips.
    """
    xp = np if isinstance(psi, np.ndarray) else get_xp()
    psi = xp.atleast_2d(psi)
    m, dim = psi.shape
    n = int(np.log2(dim))
    ZS = xp.asarray(z_signs(n))
    idx = xp.arange(dim)
    prob = xp.abs(psi) ** 2
    conj = xp.conj(psi)

    Xv = xp.empty((m, n)); Yv = xp.empty((m, n))
    for q in range(n):
        fl = idx ^ (1 << q)
        bit = (idx >> q) & 1
        Xv[:, q] = xp.real(xp.sum(conj * psi[:, fl], axis=1))
        Yv[:, q] = xp.real(xp.sum(conj * (1j * (1.0 - 2.0 * bit)) * psi[:, fl], axis=1))
    Zv = prob @ ZS

    pairs = list(combinations(range(n), 2))
    XX = xp.empty((m, len(pairs))); YY = xp.empty_like(XX); ZZ = xp.empty_like(XX)
    for k, (i, j) in enumerate(pairs):
        fl = idx ^ (1 << i) ^ (1 << j)
        bi = (idx >> i) & 1
        bj = (idx >> j) & 1
        XX[:, k] = xp.real(xp.sum(conj * psi[:, fl], axis=1))
        fac = -(1.0 - 2.0 * bi) * (1.0 - 2.0 * bj)
        YY[:, k] = xp.real(xp.sum(conj * fac * psi[:, fl], axis=1))
        ZZ[:, k] = prob @ (ZS[:, i] * ZS[:, j])
    return to_numpy(xp.hstack([Xv, Yv, Zv, XX, YY, ZZ]))


# ---------------------------------------------------------------------------
# Bank layout + observable families (unchanged)
# ---------------------------------------------------------------------------

AQFM_FAMILIES = ("X", "Y", "Z", "XYZ", "XX", "YY", "ZZ", "pairs", "full")
LOCAL_FAMILIES = {"X", "Y", "Z", "XYZ"}
PAIR_RICH_FAMILIES = {"XX", "YY", "ZZ", "pairs", "full"}


def slices_for_p(p: int):
    q = p * (p - 1) // 2
    a, k = {}, 0
    for name, d in (("X", p), ("Y", p), ("Z", p), ("XX", q), ("YY", q), ("ZZ", q)):
        a[name] = slice(k, k + d)
        k += d
    return a, k


def family_matrix(bank: np.ndarray, family: str, p: int) -> np.ndarray:
    sl, _ = slices_for_p(p)
    if family in sl:
        return bank[:, sl[family]]
    if family == "XYZ":
        return np.hstack([bank[:, sl["X"]], bank[:, sl["Y"]], bank[:, sl["Z"]]])
    if family == "pairs":
        return np.hstack([bank[:, sl["XX"]], bank[:, sl["YY"]], bank[:, sl["ZZ"]]])
    if family == "full":
        return np.hstack([bank[:, sl[k]] for k in ("X", "Y", "Z", "XX", "YY", "ZZ")])
    raise ValueError(family)


def _bank_from_product_state(loc: np.ndarray) -> np.ndarray:
    """loc: (m, p, 3) single-qubit (X, Y, Z).  For a product state
    <P_i Q_j> = <P_i><Q_j>, so XX/YY/ZZ are exact products."""
    m, p, _ = loc.shape
    pairs = list(combinations(range(p), 2))
    i_idx = np.array([i for i, _ in pairs], dtype=int)
    j_idx = np.array([j for _, j in pairs], dtype=int)
    parts = [loc[:, :, 0], loc[:, :, 1], loc[:, :, 2]]
    for k in range(3):
        parts.append(loc[:, i_idx, k] * loc[:, j_idx, k] if pairs else
                     np.empty((m, 0)))
    return np.hstack(parts)


def compute_bank(X: np.ndarray, J: float, tau: float, n_steps: int | None = None,
                 chunk: int | None = None, method: str = "rk4") -> np.ndarray:
    """Full observable bank (float32, as in the originals) for every row of X.

    method="rk4"    : FROZEN v1 protocol (400-step midpoint RK4). n_steps
                      defaults to 400.
    method="meanfield": classical product-state surrogate (see meanfield_local);
                      pair observables are products of single-qubit values.
    method="strang" : protocol v2 (unitary Strang splitting, n_steps defaults to
                      strang_steps(tau)).  At J == 0 the product-state
                      factorisation is used (exact identity, much faster).
    `chunk` bounds memory: rows are evolved `chunk` at a time.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, p = X.shape
    if method == "rk4":
        n_steps = 400 if n_steps is None else n_steps
        evolve = evolve_batch_rk4
    elif method == "meanfield":
        n_steps = strang_steps(tau) if n_steps is None else n_steps
        loc = meanfield_local(X, J, tau, n_steps)
        return _bank_from_product_state(loc).astype(np.float32)
    elif method == "strang":
        n_steps = strang_steps(tau) if n_steps is None else n_steps
        if not _is_matrix(J) and J == 0:
            loc = single_qubit_strang(X.ravel(), tau, n_steps).reshape(m, p, 3)
            return _bank_from_product_state(loc).astype(np.float32)
        evolve = evolve_batch_strang
    else:
        raise ValueError(method)
    if chunk is None:
        # ~100 MB working set on CPU, ~1 GB per array on GPU (AQFM_GPU=1)
        chunk = default_chunk(p) if evolve is evolve_batch_strang else max(1, int(6e6 // (1 << p)))
    out = []
    for s in range(0, m, chunk):
        psi = evolve(X[s:s + chunk], J, tau, n_steps)
        out.append(observables_batch(psi).astype(np.float32))
    return np.vstack(out)
