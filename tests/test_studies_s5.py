"""Tests for S5 (studies/common/dtwa.py, studies/s5_twin/run.py).

The two limits in the plan (Section 2) are checked to rounding error before anything else:
one unit-vector trajectory is mean-field, and with J = 0 the enumerated DTWA is exact."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, meanfield_local, slices_for_p, strang_steps  # noqa: E402
from studies.common import bankstore as bs  # noqa: E402
from studies.common import dtwa  # noqa: E402

_spec = importlib.util.spec_from_file_location("s5run", ROOT / "studies" / "s5_twin" / "run.py")
S5 = importlib.util.module_from_spec(_spec)
sys.modules["s5run"] = S5
_spec.loader.exec_module(S5)


@pytest.mark.parametrize("p,J,tau", [(3, 0.7, 3.0), (5, 1.0, 10.0), (4, 2.0, 1.0), (6, 4.0 / 5, 20.0)])
def test_one_unit_trajectory_is_mean_field(p, J, tau):
    X = np.random.default_rng(p).standard_normal((6, p))
    n = strang_steps(tau)
    S = dtwa.evolve(X, np.tile([1.0, 0.0, 0.0], (1, p, 1)), J, tau, n)[:, 0]
    loc = np.stack([S[..., 0], -S[..., 1], S[..., 2]], -1)          # bank convention for Y
    assert np.abs(loc - meanfield_local(X, J, tau, n)).max() < 1e-9


@pytest.mark.parametrize("p,tau", [(2, 3.0), (3, 5.0), (4, 20.0)])
def test_dtwa_is_exact_without_coupling(p, tau):
    X = np.random.default_rng(10 + p).standard_normal((5, p))
    B, Bs, M = dtwa.dtwa_bank(X, 0.0, tau, strang_steps(tau), M=4 ** p)
    assert M == 4 ** p and np.array_equal(B, Bs)                   # enumerated: no noise check
    ex = compute_bank(X, 0.0, tau, method="strang")
    assert B.shape == ex.shape and np.abs(B - ex).max() < 1e-5      # float32 banks, pairs included


def test_initial_points():
    E = dtwa.initial_points(3, 10_000)
    assert E.shape == (64, 3, 3)                                    # 4^3 enumerated
    assert np.allclose(E.mean(0), [[1, 0, 0]] * 3)
    assert np.allclose((E ** 2).sum(-1), 3.0)                       # length sqrt(3)
    R = dtwa.initial_points(8, 512, np.random.default_rng(0))
    assert R.shape == (512, 8, 3) and set(np.unique(R[..., 1:])) == {-1.0, 1.0} and (R[..., 0] == 1).all()


def test_random_sampling_converges_to_enumeration():
    p, J, tau = 4, 4.0 / 3, 5.0
    X = np.random.default_rng(3).standard_normal((4, p))
    full, _, _ = dtwa.dtwa_bank(X, J, tau, strang_steps(tau), M=256)
    S0 = dtwa.initial_points(p, 255, np.random.default_rng(1))      # 255 < 256: random draws
    samp = dtwa.bank_from_trajectories(dtwa.evolve(X, S0, J, tau, strang_steps(tau)))
    sl, _ = slices_for_p(p)
    err = np.abs(samp[:, :3 * p] - full[:, :3 * p]).max()
    assert err < 0.25                                               # ~ 1/sqrt(255) noise, a few sigma


def test_measures_limits():
    """twin == exact -> U, E ~ 0; twin == mean-field -> U_DTWA == U_MF"""
    p, c, tau = 3, 4.0, 5.0
    X = bs.pool_inputs(p)[:bs.BLOCK]
    J = S5.coupling(p, c, "Q")
    BQ = compute_bank(X, J, tau, method="strang")
    BMF = compute_bank(X, J, tau, method="meanfield")
    a = S5.per_sample(BQ, BMF, BQ, BQ, p, c, "c4_t5")
    v = S5.point_values(a)
    assert v["U_DTWA"] < 0.02 and v["E_DTWA"] < 1e-6 and abs(v["corr_match"] - 1) < 1e-6
    b = S5.point_values(S5.per_sample(BQ, BMF, BMF, BMF, p, c, "c4_t5"))
    assert np.isclose(b["U_DTWA"], b["U_MF"]) and b["U_MF"] > 0.05


def test_end_to_end_tiny(tmp_path, monkeypatch):
    """banks -> measure -> score at p = 2, 3, with fast fake step counts"""
    banks = tmp_path / "banks"
    monkeypatch.setattr(S5, "BANKS", banks)
    monkeypatch.setattr(S5, "OUT", tmp_path / "s5")
    monkeypatch.setattr(S5, "strang_steps", lambda tau: 40)
    monkeypatch.setitem(S5.DESIGN, "ps", [2, 3])
    monkeypatch.setitem(S5.DESIGN, "fit_p_min", 2)
    monkeypatch.setitem(S5.DESIGN, "test_p", 3)
    monkeypatch.setitem(S5.DESIGN, "n_boot", 20)
    for p in (2, 3):
        X = bs.pool_inputs(p)[:bs.BLOCK]
        for cl in S5.cells():
            for kind, method in (("Q", "strang"), ("MF", "meanfield")):
                B = compute_bank(X, S5.coupling(p, cl["c"], "Q"), cl["tau"], n_steps=40, method=method)
                f = bs.block_path(banks, p, kind, cl["c"], cl["tau"], 0)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, B)
    (tmp_path / "s5").mkdir()
    S5.do_banks(workers=2, M=64)
    S5.do_measure(workers=2)
    S5.do_score()
    M = pd.read_csv(tmp_path / "s5" / "S5_MEASURES.csv")
    assert len(M) == 2 * 14 and (M.M_used == np.where(M.p == 2, 16, 64)).all()
    K = pd.read_csv(tmp_path / "s5" / "S5_CHECKS.csv")
    assert list(K.check) == ["D0", "D1", "D2", "D3"]
    assert len(pd.read_csv(tmp_path / "s5" / "S5_NOISE_FLAGS.csv")) == 0   # p <= 3: enumerated
