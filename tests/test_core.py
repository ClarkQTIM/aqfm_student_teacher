import numpy as np

from aqfm_prescreen import dial
from aqfm_prescreen.evaluate import fold_splits, preprocess_fold_inputs


def test_classical_candidates_respect_declared_dimension():
    X, y = dial.generate("local", 0.5, 0, n=200)
    tr, te = fold_splits(X, y)[0]
    Ztr, Zte = preprocess_fold_inputs(X, tr, te)
    for nm, D, fn in dial.classical_candidates(6, "v2"):
        out = fn(Ztr, y[tr], Zte)
        if out is None:
            continue
        assert out[0].shape == (len(tr), D), nm
        assert out[1].shape == (len(te), D), nm


def test_meanfield_equals_exact_at_J0_and_weak_coupling():
    from aqfm_prescreen.aqfm import compute_bank
    rng = np.random.default_rng(0)
    Z = np.clip(rng.standard_normal((50, 4)), -3, 3)
    a = compute_bank(Z, 0.0, 2.0, method="strang")
    b = compute_bank(Z, 0.0, 2.0, method="meanfield")
    assert np.abs(a - b).max() < 1e-5
    e = compute_bank(Z, 0.02, 2.0, method="strang")[:, :12]
    m = compute_bank(Z, 0.02, 2.0, method="meanfield")[:, :12]
    assert np.corrcoef(e.ravel(), m.ravel())[0, 1] > 0.999


def test_matrix_coupling_matches_uniform_scalar():
    from aqfm_prescreen.aqfm import compute_bank
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 4)); J = 0.37
    Jm = J * (np.ones((4, 4)) - np.eye(4))
    for method in ("strang", "meanfield"):
        a = compute_bank(X, J, 1.3, method=method)
        b = compute_bank(X, Jm, 1.3, method=method)
        assert np.allclose(a, b, atol=1e-6)


def test_matrix_coupling_structured_only_couples_listed_pairs():
    from aqfm_prescreen.aqfm import compute_bank, slices_for_p
    X = np.array([[0.5, -1.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, -1.0], [0.5, 0.0, 1.0]])
    Jm = np.zeros((3, 3)); Jm[0, 1] = Jm[1, 0] = 1.0
    z1 = compute_bank(X, Jm, 2.0, method="strang")[:, slices_for_p(3)[0]["Z"]][:, 0]
    assert abs(z1[0] - z1[1]) > 1e-3          # qubit 1 feels x2 ...
    assert abs(z1[2] - z1[3]) < 1e-9          # ... but not x3


def test_teacher_student_teachers():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import teacher_student as TS
    d = dict(TS.DESIGN, n=200)
    sp = {s["name"]: s for s in TS.teacher_specs(dict(d, ps=[4]))}
    _, y0, m0 = TS.make_dataset(sp["ts_J0_p4_k0_t1_local_s0"], d)
    _, _, mm = TS.make_dataset(sp["ts_MF_p4_k2_t1_local_s0"], d)
    _, _, mq = TS.make_dataset(sp["ts_Q_p4_k2_t1_local_s0"], d)
    assert abs(m0["entanglement"]) < 1e-6 and abs(mm["entanglement"]) < 1e-6   # product states are pure
    assert mq["entanglement"] > 0.01                                            # exact coupled states are not
    assert 0.2 < y0.mean() < 0.8 and m0["oracle_auc"] > 0.8
