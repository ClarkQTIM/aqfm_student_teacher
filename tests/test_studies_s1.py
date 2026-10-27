"""Tests for S1 (studies/s1_scaling/run.py): teachers, the Hermite design, Part A."""
import importlib.util
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.common import bankstore as bs  # noqa: E402

_spec = importlib.util.spec_from_file_location("s1run", ROOT / "studies" / "s1_scaling" / "run.py")
S1 = importlib.util.module_from_spec(_spec)
sys.modules["s1run"] = S1                 # so worker processes can find its functions
_spec.loader.exec_module(S1)


def test_teacher_specs():
    T = S1.teacher_specs(6)
    assert len(T) == 14 * 2 * 3 + 5 * 3
    assert len({t["name"] for t in T}) == len(T)
    j0 = [t for t in T if t["kind"] == "J0"]
    assert {t["tau"] for t in j0} == {1.0, 3.0, 5.0, 10.0, 20.0}
    assert {t["region"] for t in j0 if t["tau"] == 1.0} == {"control: short anneal"}
    assert len(S1.bank_keys(6)) == 14 * 2 + 5


def test_d_max():
    assert S1.d_max(12) == 4 and S1.d_max(8) == 5 and S1.d_max(2) == 12
    for p in (2, 3, 4, 5, 6, 8, 10, 12):
        assert comb(p + S1.d_max(p), S1.d_max(p)) - 1 <= S1.DESIGN["max_terms"]


def test_hermite_design_is_degree_ordered_and_nearly_orthonormal():
    Z = np.random.default_rng(0).standard_normal((200_000, 3))
    X = S1.hermite_design(Z, 3)
    assert X.shape[1] == comb(6, 3)
    G = X.T @ X / len(X)
    assert np.abs(G - np.eye(len(G))).max() < 0.05
    X2 = S1.hermite_design(Z[:50], 2)
    assert np.allclose(S1.hermite_design(Z[:50], 3)[:, :X2.shape[1]], X2)    # prefix property


def test_degree_profile_finds_the_true_degree():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((6000, 4))
    s = (Z[:, 0] ** 2 - 1) + Z[:, 1] * Z[:, 2]                 # degree exactly 2
    X = S1.hermite_design(Z, 3)
    prof = S1.degree_profile(X[:5000], s[:5000, None], X[5000:], s[5000:, None], 4, 3)
    assert prof[1][0] < 0.05 and prof[2][0] > 0.999999 and prof[3][0] > 0.999999


def test_end_to_end_tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "BLOCK", 20)
    monkeypatch.setattr(S1, "BANKS", tmp_path / "banks")
    monkeypatch.setattr(S1, "OUT", tmp_path / "s1")
    monkeypatch.setitem(S1.DESIGN, "n_blocks", 3)
    monkeypatch.setitem(S1.DESIGN, "train_rows", 45)
    monkeypatch.setitem(S1.DESIGN, "max_terms", 220)       # so degree 4 exists at every p
    monkeypatch.setitem(S1.DESIGN, "fit_ps", [4, 5, 6])
    monkeypatch.setitem(S1.DESIGN, "n_boot", 50)
    (tmp_path / "s1").mkdir()
    ps = [2, 4, 5, 6]
    S1.do_banks(ps, gpus=0, cpu_workers=4)
    S1.do_teachers(ps, workers=4)
    M = pd.read_csv(tmp_path / "s1" / "S1_TEACHERS.csv")
    assert len(M) == len(ps) * 99 and M.pos_rate.between(0, 1).all()
    T = np.load(tmp_path / "s1" / "teachers" / "p4.npz")
    assert T["S"].shape == (60, 99) and np.allclose(T["S"].mean(0), 0, atol=1e-5)
    S1.do_partA(ps, workers=4)
    A = pd.read_csv(tmp_path / "s1" / "S1_PARTA.csv")
    assert set(A.p) == set(ps) and A.R2.notna().all()
    S1.do_scoreA()
    K = pd.read_csv(tmp_path / "s1" / "S1_PARTA_CHECKS.csv")
    assert list(K.check) == ["A1", "A2", "A2", "A3"] * 2
    S1.do_banks(ps, gpus=0, cpu_workers=2)                         # restartable: nothing left to do
