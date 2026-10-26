"""Tests for the shared bank store (studies/common) and S2 (studies/s2_meanfield/run.py)."""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, slices_for_p  # noqa: E402
from studies.common import bankstore as bs  # noqa: E402
from studies.common.cells import cells, coupling  # noqa: E402

_spec = importlib.util.spec_from_file_location("s2run", ROOT / "studies" / "s2_meanfield" / "run.py")
S2 = importlib.util.module_from_spec(_spec)
sys.modules["s2run"] = S2                 # so worker processes can find its functions
_spec.loader.exec_module(S2)


# ---------------------------------------------------------------- cells and pool

def test_cells_and_coupling():
    C = cells()
    assert len(C) == 14 and sum(c["region"] == "window" for c in C) == 12
    assert coupling(8, 4.0, "Q") == pytest.approx(4 / 7) and coupling(8, 4.0, "J0") == 0.0


def test_pool_is_fixed_clipped_and_independent_of_use():
    a, b = bs.pool_inputs(5), bs.pool_inputs(5)
    assert a.shape == (bs.POOL_ROWS, 5) and np.array_equal(a, b)
    assert np.abs(a).max() <= 3.0
    assert not np.array_equal(bs.pool_inputs(4)[:, :4], a[:, :4])          # different p, different pool


def test_pieces_cover_the_block_exactly():
    for p in (4, 12, 14, 16):
        sp = bs.piece_specs("x", p, "Q", 2.0, 5.0, 3)
        assert sp[0][0] == 3 * bs.BLOCK and sp[-1][1] == 4 * bs.BLOCK
        assert all(e == s2 for (_, e), (s2, _) in zip(sp, sp[1:]))


def test_pieces_merge_to_the_direct_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "BLOCK", 12)
    monkeypatch.setattr(bs, "piece_rows", lambda p: 5)                       # uneven last piece
    for s, e in bs.piece_specs(tmp_path, 3, "Q", 2.0, 3.0, 1):
        assert bs.compute_piece((tmp_path, 3, "Q", 2.0, 3.0, 1, s, e))[0].startswith("ok")
    assert bs.merge_block(tmp_path, 3, "Q", 2.0, 3.0, 1)
    got = np.load(bs.block_path(tmp_path, 3, "Q", 2.0, 3.0, 1))
    want = compute_bank(bs.pool_inputs(3)[12:24], coupling(3, 2.0, "Q"), 3.0, method="strang")
    assert np.allclose(got, want)
    assert list(bs.bank_dir(tmp_path, 3, "Q", 2.0, 3.0).glob("*rows*")) == []   # pieces cleaned up
    assert bs.compute_piece((tmp_path, 3, "Q", 2.0, 3.0, 1, 12, 17))[0] == "skip"


def test_exact_and_meanfield_agree_at_tiny_coupling():
    """sign conventions of the two simulators match (Y especially): differences ~ J"""
    Z = bs.pool_inputs(4)[:20]
    sl, _ = slices_for_p(4)
    Q = compute_bank(Z, 1e-6, 5.0, method="strang").astype(float)
    M = compute_bank(Z, 1e-6, 5.0, method="meanfield").astype(float)
    for k in "XYZ":
        assert np.abs(Q[:, sl[k]] - M[:, sl[k]]).max() < 1e-4


# ---------------------------------------------------------------- the measures

def _bank(p, c, tau, kind, m=60):
    return compute_bank(bs.pool_inputs(p)[:m], coupling(p, c, "Q"), tau,
                        method="meanfield" if kind == "MF" else "strang").astype(float)


def test_measures_vanish_when_meanfield_is_the_truth():
    p = 4
    BM = _bank(p, 4.0, 5.0, "MF")
    a = S2.per_sample(BM, BM, p, 4.0, "c4_t5", n_weights=5)
    v = S2.point_values(a)
    assert v["E"] == 0
    assert v["C_pair"] < 1e-6 and v["C_coll"] < 1e-6              # a product state has no correlations
    assert v["U"] < 1e-10                                         # the score IS linear in these features


def test_connected_correlations_and_collective_sum_by_hand():
    p = 3
    BQ = _bank(p, 4.0, 5.0, "Q", m=5)
    a = S2.per_sample(BQ, _bank(p, 4.0, 5.0, "MF", m=5), p, 4.0, "c4_t5", n_weights=3, folds=2)
    sl, _ = slices_for_p(p)
    Z, ZZ = BQ[:, sl["Z"]], BQ[:, sl["ZZ"]]
    C = np.zeros((5, p, p))
    for k, (i, j) in enumerate(combinations(range(p), 2)):
        C[:, i, j] = C[:, j, i] = ZZ[:, k] - Z[:, i] * Z[:, j]
    J = coupling(p, 4.0, "Q")
    want = J * np.abs(C.sum(2)).mean(1)
    assert np.allclose(a["ccoll"], want)
    assert np.allclose(a["czz"], np.abs(C[:, [0, 0, 1], [1, 2, 2]]).mean(1))


def test_fit_alpha_recovers_a_power_law():
    ps = np.array([4, 6, 8, 12, 16])
    al, a = S2.fit_alpha(ps, 3.0 * ps ** -0.7)
    assert al == pytest.approx(0.7) and np.exp(a) == pytest.approx(3.0)


# ---------------------------------------------------------------- end to end, tiny

def test_banks_measure_score_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "BLOCK", 20)
    monkeypatch.setattr(S2, "BANKS", tmp_path / "banks")
    monkeypatch.setattr(S2, "OUT", tmp_path / "s2")
    monkeypatch.setitem(S2.DESIGN, "n_boot", 40)
    monkeypatch.setitem(S2.DESIGN, "n_weights", 4)
    (tmp_path / "s2").mkdir()
    ps = [2, 4, 5, 6]
    S2.do_banks(ps, workers=4)
    assert S2.available_ps() == ps
    S2.do_measure(workers=4)
    M = pd.read_csv(tmp_path / "s2" / "S2_MEASURES.csv")
    assert len(M) == len(ps) * 14 and M[S2.MEASURES].notna().all().all()
    S2.do_score()
    A = pd.read_csv(tmp_path / "s2" / "S2_ALPHAS.csv")
    assert len(A) == 14 * len(S2.MEASURES)
    assert (A.alpha_lo <= A.alpha_hi).all()
    K = pd.read_csv(tmp_path / "s2" / "S2_CHECKS.csv")
    assert list(K.check) == ["M1", "M2", "M3", "M4"]
    S2.do_banks(ps, workers=2)                                      # restartable: everything skipped
