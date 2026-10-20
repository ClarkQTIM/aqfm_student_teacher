"""Port-fidelity tests: the batched implementation must reproduce the ORIGINAL
single-sample code (tests/reference_original.py, copied verbatim) to round-off."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import reference_original as ref  # noqa: E402

from aqfm_prescreen.aqfm import (evolve_batch_rk4, evolve_exact,  # noqa: E402
                                 observables_batch)


@pytest.mark.parametrize("p,J,tau", [(2, 0.0, 1.0), (3, 1.0, 2.0), (4, 1.0, 10.0),
                                     (5, 0.0, 5.0)])
def test_batched_rk4_matches_original(p, J, tau):
    rng = np.random.default_rng(p)
    X = rng.uniform(-3, 3, size=(7, p))
    ZS = ref.z_signs(p)
    batch = observables_batch(evolve_batch_rk4(X, J, tau, 400))
    orig = np.array([ref.observables(ref.evolve_one(x, J, tau, 400, ZS), ZS)
                     for x in X])
    assert np.max(np.abs(batch - orig)) < 1e-10


@pytest.mark.parametrize("p,J,tau", [(2, 1.0, 10.0), (3, 1.0, 10.0), (3, 0.0, 10.0)])
def test_rk4_close_to_exact_propagator(p, J, tau):
    """RK4 (400 steps) vs exact per-step exponentials (400 steps, same midpoint
    discretisation) and vs a 4000-step exact reference (continuum proxy)."""
    rng = np.random.default_rng(10 + p)
    X = rng.uniform(-3, 3, size=(4, p))
    rk = observables_batch(evolve_batch_rk4(X, J, tau, 400))
    ex400 = observables_batch(np.array([evolve_exact(x, J, tau, 400) for x in X]))
    ex4k = observables_batch(np.array([evolve_exact(x, J, tau, 4000) for x in X]))
    assert np.max(np.abs(rk - ex400)) < 1e-4      # RK4 truncation error
    assert np.max(np.abs(ex400 - ex4k)) < 1e-2    # time-discretisation error


# ---------------------------------------------------------------------------
# Protocol v2 integrator
# ---------------------------------------------------------------------------
from aqfm_prescreen.aqfm import (compute_bank, evolve_batch_strang,  # noqa: E402
                                 strang_steps)


def _strang_flip_reference(X, J, tau, n_steps):
    """Straightforward index-gather Strang (the first implementation), kept as
    a reference for the reshape-based fast version."""
    from aqfm_prescreen.aqfm import _flip_index, data_diagonal_batch, z_signs
    X = np.atleast_2d(X); m, n = X.shape
    dd = data_diagonal_batch(X, J, z_signs(n))
    psi = np.full((m, 1 << n), 1 / np.sqrt(1 << n), dtype=complex)
    dt = tau / n_steps
    for k in range(n_steps):
        s = (k + .5) / n_steps
        h = np.exp(-.5j * dt * s * dd); psi *= h
        c, sn = np.cos(dt * (1 - s)), 1j * np.sin(dt * (1 - s))
        for f in _flip_index(n):
            psi = c * psi + sn * psi[:, f]
        psi *= h
    return psi


@pytest.mark.parametrize("p,J,tau", [(3, 1.0, 10.0), (5, 0.0, 2.0)])
def test_strang_reshape_equals_gather(p, J, tau):
    X = np.random.default_rng(1).uniform(-3, 3, (4, p))
    a = evolve_batch_strang(X, J, tau, 300)
    b = _strang_flip_reference(X, J, tau, 300)
    assert np.max(np.abs(a - b)) < 1e-12


@pytest.mark.parametrize("p,tau", [(2, 0.3), (4, 10.0), (6, 2.0)])
def test_j0_factorised_bank_equals_statevector(p, tau):
    X = np.random.default_rng(2).uniform(-3, 3, (5, p))
    fast = compute_bank(X, 0.0, tau, method="strang")
    full = observables_batch(evolve_batch_strang(X, 0.0, tau, strang_steps(tau)))
    assert np.max(np.abs(fast - full)) < 1e-5     # float32 storage


@pytest.mark.parametrize("p,J,tau", [(4, 1.0, 10.0), (8, 1.0, 10.0), (8, 0.0, 10.0)])
def test_strang_v2_close_to_exact(p, J, tau):
    """v2 (Strang, strang_steps) vs exact piecewise propagation, fine grid."""
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    from aqfm_prescreen.aqfm import _flip_index, data_diagonal, z_signs
    x = np.clip(np.random.default_rng(3).standard_normal(p), -3, 3)
    dim = 1 << p
    r = np.concatenate([np.arange(dim)] * p)
    c = np.concatenate(_flip_index(p))
    Hi = sparse.csr_matrix((-np.ones(p * dim), (r, c)), shape=(dim, dim))
    Hd = sparse.diags(data_diagonal(x, J, z_signs(p)))
    psi = np.full(dim, 1 / np.sqrt(dim), dtype=complex); N = 2000; dt = tau / N
    for k in range(N):
        s = (k + .5) / N
        psi = expm_multiply((-1j * dt) * ((1 - s) * Hi + s * Hd), psi)
    ref = observables_batch(psi[None])
    got = compute_bank(x[None], J, tau, method="strang")
    assert np.max(np.abs(got - ref)) < 1e-4
