import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import ts_tau_sweep as T  # noqa: E402
from aqfm_prescreen.aqfm import compute_bank, data_diagonal_batch, family_matrix  # noqa: E402


def test_ground_state_is_argmin():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((50, 4))
    J = 0.7
    G = T.ground_state_bank(Z, J)
    sig = family_matrix(G, "Z", 4)
    ZS = T.spin_basis(4)
    E = data_diagonal_batch(Z, J, ZS)
    e_sig = -(Z * sig).sum(1) - J * 0.5 * (sig.sum(1) ** 2 - 4)
    assert np.allclose(e_sig, E.min(1))
    assert np.allclose(family_matrix(G, "XYZ", 4)[:, :8], 0)          # X, Y vanish


def test_slow_anneal_reaches_ground_state():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((40, 3))
    J = 2.0 / 2
    G = family_matrix(T.ground_state_bank(Z, J), "Z", 3)
    B = family_matrix(compute_bank(Z, J, 60.0, method="strang"), "Z", 3)
    assert (np.sign(B) == G).all(1).mean() >= 0.95
    assert np.abs(B - G).mean() < 0.15


def test_design_well_specified():
    d = T.DESIGN
    assert set(d["teacher_taus"]) <= set(d["student_taus"])
    assert set(d["kappas"]) <= set(d["student_kappas"])
    assert len(T.teacher_specs()) == 864


def test_scaling_design():
    import ts_scaling as S
    assert S.DESIGN is T.DESIGN                     # workers see the same design
    assert len(S.teacher_specs()) == 625
    assert S.DESIGN["teacher_taus"] == S.DESIGN["student_taus"]
    assert 1.5 in S.DESIGN["kappas"] and max(S.DESIGN["curve_mults"]) == 64


def test_shot_noise_statistics():
    import ts_addons as A
    v = np.array([[-0.9, -0.3, 0.0, 0.5, 1.0]])
    B = np.repeat(v, 20000, axis=0)
    for S in (100, 1000):
        N = A.shot_noisy(B, S, "test")
        assert np.allclose(N.mean(0), v[0], atol=0.01)                    # unbiased
        assert np.allclose(N.var(0), (1 - v[0] ** 2) / S, rtol=0.1, atol=1e-6)   # binomial variance
        assert N.min() >= -1 and N.max() <= 1


def test_addon_folds_match_main_run():
    import ts_addons as A
    import teacher_student as ts
    rng = np.random.default_rng(0)
    Z, y = rng.standard_normal((600, 4)), rng.integers(0, 2, 600)
    from sklearn.model_selection import StratifiedKFold
    main = list(StratifiedKFold(ts.DESIGN["outer"], shuffle=True, random_state=ts.DESIGN["cv_seed"]).split(Z, y))
    for (a, b), (c, d) in zip(main, A._folds(Z, y)):
        assert (a == c).all() and (b == d).all()
