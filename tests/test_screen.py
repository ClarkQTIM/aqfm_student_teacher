"""Tests for experiments/ts_screen.py: the non-AQFM teachers and the noisy copies."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import teacher_student as ts  # noqa: E402
import ts_screen as SC  # noqa: E402
import ts_tau_sweep as tt  # noqa: E402
from aqfm_prescreen.aqfm import family_matrix  # noqa: E402


# ---------------------------------------------------------------- design sizes

def test_block_sizes_match_plan():
    F, N = SC.foreign_specs(), SC.noise_specs()
    assert len(F) == 162                               # 9 families x 3 p x 3 seeds x 2 T
    assert len(N) == 126                               # 108 exact/mean-field + 18 J=0
    assert sum(s["kind"] == "J0" for s in N) == 18
    names = [s["name"] for s in F + N]
    assert len(set(names)) == len(names)               # unique
    assert not set(names) & {s["name"] for s in tt.teacher_specs()}   # never clash with the sweep


def test_noise_copies_keep_the_original_seed():
    orig = {s["name"]: s for s in tt.teacher_specs()}
    for s in SC.noise_specs():
        o = orig[s["base"]]
        assert s["data_seed"] == o["data_seed"] and s["temp"] == 1.0
        assert {k: s[k] for k in ("kind", "p", "kappa", "tau", "readout", "seed")} == \
               {k: o[k] for k in ("kind", "p", "kappa", "tau", "readout", "seed")}


# ---------------------------------------------------------------- pairing in T

def _monotone_pair(score, y_cold, y_hot):
    """same uniforms u, y = u < sigmoid(score/T): a hotter T moves every probability
    toward 1/2, so for score > 0 a hot 1 implies a cold 1, and for score < 0 the
    other way round"""
    pos, neg = score > 0, score < 0
    return (y_hot[pos] <= y_cold[pos]).all() and (y_hot[neg] >= y_cold[neg]).all()


def test_foreign_temperatures_are_paired():
    F = {s["name"]: s for s in SC.foreign_specs()}
    for fam in SC.DESIGN["foreign_families"]:
        a, b = F[f"fr_{fam}_p6_s1_T0.25"], F[f"fr_{fam}_p6_s1_T1"]
        ya, sa, _ = SC.make_foreign(a)
        yb, sb, _ = SC.make_foreign(b)
        assert np.array_equal(sa, sb)                  # same s(z)
        assert _monotone_pair(sa, ya, yb)
        assert (ya != yb).any()                        # but the hot copy really is noisier


def test_foreign_is_deterministic():
    s = SC.foreign_specs()[17]
    y1, s1, m1 = SC.make_foreign(s)
    y2, s2, m2 = SC.make_foreign(s)
    assert np.array_equal(y1, y2) and np.array_equal(s1, s2) and m1["structure"] == m2["structure"]


# ---------------------------------------------------------------- what each family is

def _r2(A, v):
    A = np.column_stack([np.ones(len(v)), A])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return 1 - np.var(v - A @ coef) / np.var(v)


def _prods(Z, k):
    from itertools import combinations
    return np.column_stack([np.prod(Z[:, list(c)], axis=1) for c in combinations(range(Z.shape[1]), k)])


@pytest.mark.parametrize("p", [4, 8])
def test_interaction_orders(p):
    """LIN is all linear; PAIR all two-way; TRIPLE all three-way (checked by least
    squares on the inputs, all pairwise products, all triple products).
    Checked on a LARGE sample: this tests the definitions.  On the real n = 600
    inputs, chance correlations of the heavy-tailed products let linear + pairwise
    terms explain ~10% of a TRIPLE score (a finite-sample effect, not a bug)."""
    Z = np.clip(np.random.default_rng(0).standard_normal((20000, p)), -3, 3)
    rng = lambda: np.random.default_rng(123)
    lin, two, three = Z, _prods(Z, 2), _prods(Z, 3)
    s, _ = SC.foreign_score("LIN", Z, rng())
    assert _r2(lin, s) > 0.999
    s, _ = SC.foreign_score("PAIR", Z, rng())
    assert _r2(two, s) > 0.999 and _r2(lin, s) < 0.05
    s, _ = SC.foreign_score("TRIPLE", Z, rng())
    assert _r2(three, s) > 0.999 and _r2(np.hstack([lin, two]), s) < 0.1


def test_netd_has_no_linear_part():
    """NETD: the linear part is removed exactly on the real inputs"""
    for p in (4, 8):
        _, Z = tt.inputs(p)
        s, _ = SC.foreign_score("NETD", Z, np.random.default_rng(7))
        assert _r2(Z, s) < 1e-10 and np.std(s) > 0


def test_parity_has_no_signal_in_any_single_input():
    """PARp: no single input predicts the score (its signal lives only in all p jointly)"""
    _, Z = tt.inputs(8)
    s, _ = SC.foreign_score("PARp", Z, np.random.default_rng(5))
    for j in range(8):
        assert abs(np.corrcoef(Z[:, j], s)[0, 1]) < 0.12
        assert abs(np.corrcoef(np.abs(Z[:, j]), s)[0, 1]) < 0.12


def test_noise_teacher_ceiling():
    F = [s for s in SC.foreign_specs() if s["kind"] == "NOISE"]
    for sp in F[:4]:
        y, score, meta = SC.make_foreign(sp)
        assert meta["ceiling"] == 0.5
        _, Z = tt.inputs(sp["p"])
        assert np.abs(np.corrcoef(np.column_stack([Z, score]).T)[-1, :-1]).max() < 0.15


# ---------------------------------------------------------------- noisy AQFM copy, end to end

def test_noise_copy_paired_with_original(tmp_path):
    """build the three banks one exact teacher needs (p = 4), make the original
    (T = 0.25) and the copy (T = 1.0) labels, and check they are paired"""
    (tmp_path / "banks" / "p4").mkdir(parents=True)
    for kind, kap, tau in (("Q", 2.0, 5.0), ("MF", 2.0, 5.0), ("GS", 2.0, 0.0)):
        tt._bank_job((tmp_path, 4, kind, kap, tau))
    name = "tt_Q_p4_k2_t5_local_s0"
    orig = {s["name"]: s for s in tt.teacher_specs()}[name]
    copy = {s["base"]: s for s in SC.noise_specs()}[name]
    y_cold, _ = tt.make_labels(tmp_path, orig)
    y_hot, meta = SC.make_noise_copy(tmp_path, copy)
    # recompute the teacher score the same way make_labels does
    rng = np.random.default_rng(orig["data_seed"])
    F = tt.readout_matrix(tt.load_bank(tmp_path, 4, "Q", 2.0, 5.0), "local", 4)
    score = tt._score_from(F, rng.standard_normal(F.shape[1]))
    assert _monotone_pair(score, y_cold, y_hot)
    assert meta["temp"] == 1.0 and abs(meta["ceiling"] - 0.74) < 0.06    # about 0.74 at T = 1


# ---------------------------------------------------------------- screen measures

def test_measures_never_see_the_map():
    """screen_measures may only import general libraries: not the map, not the
    teacher code, not anything that could load a bank or a teacher score"""
    import ast
    src = (Path(__file__).resolve().parents[1] / "experiments" / "screen_measures.py").read_text()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "warnings", "itertools", "hashlib", "numpy", "scipy", "sklearn"}, mods
    import inspect
    import screen_measures as SM
    assert list(inspect.signature(SM.all_measures).parameters) == ["Z", "y", "name", "folds"]


def _foreign(nm):
    sp = {s["name"]: s for s in SC.foreign_specs()}[nm]
    y, _, _ = SC.make_foreign(sp)
    _, Z = tt.inputs(sp["p"])
    return Z, y


@pytest.mark.parametrize("nm,order", [("fr_LIN_p6_s0_T0.25", "1"), ("fr_PAIR_p6_s0_T0.25", "2"),
                                      ("fr_TRIPLE_p6_s0_T0.25", "3"), ("fr_NOISE_p6_s0_T0.25", "none found")])
def test_interaction_ladder_on_known_orders(nm, order):
    import screen_measures as SM
    Z, y = _foreign(nm)
    assert SM.measure_F1(Z, y, SM.outer_folds(Z, y))["F1_order"] == order


def test_nn_bound_on_noise_and_structure():
    import screen_measures as SM
    Z, y = _foreign("fr_NOISE_p6_s0_T0.25")
    assert SM.measure_L3(Z, y)["L3_ceiling_acc"] < 0.56          # nothing to learn: about 0.5
    Z, y = _foreign("fr_LIN_p4_s0_T0.25")
    assert SM.measure_L3(Z, y)["L3_ceiling_acc"] > 0.75


def test_measures_deterministic_and_complete():
    import screen_measures as SM
    Z, y = _foreign("fr_NETD_p4_s0_T0.25")
    a = SM.all_measures(Z, y, "x")
    b = SM.all_measures(Z, y, "x")
    assert a == b or all((a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]) for k in a)   # NaN-safe
    for k in ("L1_slope", "L1_asymptote", "L2_gain", "L3_ceiling_acc", "F1_order"):
        assert k in a


# ---------------------------------------------------------------- evaluation (pure part)

def _fake_library(n=240, seed=0):
    """a synthetic per-dataset table where the screen works by construction"""
    import pandas as pd
    import screen_evaluate as SE
    rng = np.random.default_rng(seed)
    blocks = np.array(["tau_sweep", "aqfm_noise", "foreign"])[rng.integers(0, 3, n)]
    D = pd.DataFrame(dict(name=[f"d{i}" for i in range(n)], base=[f"b{i // 2}" for i in range(n)],
                          block=blocks, p=rng.choice([4, 6, 8], n), temp=rng.choice([0.25, 1.0], n),
                          kind=np.where(blocks == "foreign", rng.choice(["LIN", "PAIR", "NOISE"], n), "Q"),
                          region=np.where(blocks == "foreign", "foreign", "other")))
    hard = (rng.random(n) < 0.4) & (D.kind != "NOISE").to_numpy()     # noise teachers have nothing to resist
    D["ceiling"] = np.where(D.kind == "NOISE", 0.5, 0.9)
    for c in SE.OPPONENTS:
        D[c] = np.where(hard, 0.7, D.ceiling - 0.01) - rng.uniform(0, 0.01, n)
    D["auc_S_Q"] = np.where(hard & (D.block != "foreign"), 0.88, D[SE.OPPONENTS].max(axis=1))
    D["L1_slope"] = np.where(hard, 0.05, 0.0) + rng.normal(0, 0.005, n)
    D["L2_gain"] = np.where(hard, 0.02, 0.0) + rng.normal(0, 0.005, n)
    return SE.add_derived(D)


def test_evaluate_definitions():
    D = _fake_library()
    assert (D.resistant.astype(bool) == (D.ceiling - D.best_cheap >= 0.05)).all()
    assert np.allclose(D.margin, D.auc_S_Q - D[["auc_classical_best", "auc_nonlinear_best", "auc_S_MF",
                                                "auc_S_J0", "auc_S_GS"]].max(axis=1))
    assert D.screen.between(0, 1).all()


def test_checks_on_a_library_where_the_screen_works():
    import screen_evaluate as SE
    R, D = SE.checks(_fake_library())
    v = R.set_index("what").value
    assert v["screen score vs 'resistant', all datasets"] > 0.95
    assert v["AQFM helps | resistant, non-AQFM teachers"] == 0.0           # by construction
    assert v["AQFM helps | resistant, AQFM-made teachers"] == 1.0
    assert R[R.check == "P9"].value.iloc[0] < 0.1                            # NOISE rarely flagged


def test_lopo_threshold_never_sees_the_test_p():
    import screen_evaluate as SE
    D = _fake_library()
    f1 = SE.lopo_flags(D)
    D2 = D.copy()
    D2.loc[D2.p == 8, "resistant"] = 1.0 - D2.loc[D2.p == 8, "resistant"]   # flip p = 8's truth
    f2 = SE.lopo_flags(D2)
    assert (f1[D.p == 8] == f2[D.p == 8]).all()          # p = 8 flags depend only on p = 4, 6


def test_auroc_handles_missing():
    import screen_evaluate as SE
    assert SE.auroc([0.1, 0.9, np.nan], [0, 1, 1]) == 1.0
    assert np.isnan(SE.auroc([0.1, 0.2], [1, 1]))
