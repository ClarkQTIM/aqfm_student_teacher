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
