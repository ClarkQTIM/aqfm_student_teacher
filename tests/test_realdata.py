"""Tests for the real-data runner (experiments/realdata.py, real_datasets.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import realdata as RL  # noqa: E402
import real_datasets as RD  # noqa: E402


def _fake_table(n=240, k=10, seed=0, groups=True):
    """two eyes per 'patient'; label driven by features 0 and 3"""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"f{i}" for i in range(k)])
    s = X.f0 + 0.8 * X[f"f{min(3, k - 1)}"] + 0.3 * rng.standard_normal(n)
    y = (s > 0).astype(int).to_numpy()
    g = np.repeat(np.arange(n // 2), 2) if groups else None
    return X, y, g


# ---------------------------------------------------------------- cleaning and splitting

def test_clean_features():
    X = pd.DataFrame({"num": [1.0, 2, 3, 4, 5], "mostly_missing": [np.nan, np.nan, 1, np.nan, np.nan],
                      "sex": ["M", "F", "F", "M", "F"], "eye": ["OD", "OS", "OD", "OS", "OD"],
                      "lens": ["phakic", "pseudo", "aphakic", "phakic", "pseudo"]})
    C, dropped = RL.clean_features(X, 0.20)
    assert "mostly_missing" in dropped
    assert "num" in C and C["sex=M"].tolist() == [1, 0, 0, 1, 0]
    assert {"lens=aphakic", "lens=phakic", "lens=pseudo"} <= set(C)        # >2 levels -> one-hot
    assert C.notna().all().all()


def test_split_is_grouped_stratified_and_deterministic():
    _, y, g = _fake_table()
    d1, d2 = RL.split_halves(y, g, "x"), RL.split_halves(y, g, "x")
    assert (d1 == d2).all()
    for patient in np.unique(g):                                           # no patient in both halves
        assert len(set(d1[g == patient])) == 1
    assert abs(d1.mean() - 0.5) < 0.05 and abs(y[d1].mean() - y[~d1].mean()) < 0.08


def test_outer_folds_keep_patients_together():
    _, y, g = _fake_table()
    for tr, te in RL.outer_folds(y, g):
        assert not set(g[tr]) & set(g[te])
    for tr, te in RL.inner_folds(y[:120], g[:120]):
        assert not set(g[:120][tr]) & set(g[:120][te])


# ---------------------------------------------------------------- prepare

def _write_generic(root, name, X, y, g):
    d = root / name
    d.mkdir(parents=True)
    D = X.copy()
    D["label"] = y
    if g is not None:
        D["group"] = g
    D.to_csv(d / "clean.csv", index=False)


def test_prepare_uses_only_the_discovery_half(tmp_path, monkeypatch):
    X, y, g = _fake_table(k=12)
    monkeypatch.setattr(RD, "DATA_ROOT", tmp_path / "data")
    _write_generic(tmp_path / "data", "toy", X, y, g)
    info = RL.prepare("toy", tmp_path / "out")
    assert info["status"] == "ok" and info["p"] == 8 and {"f0", "f3"} <= set(info["qubits"])
    Z, Xall, yy, disc, groups, _ = (lambda P: (P["Z"], P["Xall"], P["y"], P["disc"], P["groups"], None))(
        np.load(tmp_path / "out" / "toy" / "prepared.npz"))
    assert np.allclose(Z[disc].mean(0), 0, atol=0.05) and np.abs(Z).max() <= 3.0
    # scrambling the CONFIRMATION half must not change which features become qubits
    X2 = X.copy()
    X2.loc[~disc] = np.random.default_rng(9).standard_normal((int((~disc).sum()), X.shape[1])) * 50
    (tmp_path / "data2").mkdir()
    _write_generic(tmp_path / "data2", "toy", X2, y, g)
    monkeypatch.setattr(RD, "DATA_ROOT", tmp_path / "data2")
    info2 = RL.prepare("toy", tmp_path / "out2")
    assert info2["qubits"] == info["qubits"]


def test_prepare_drops_too_few_features(tmp_path, monkeypatch):
    X, y, g = _fake_table(k=3)
    monkeypatch.setattr(RD, "DATA_ROOT", tmp_path)
    _write_generic(tmp_path, "tiny", X, y, g)
    info = RL.prepare("tiny", tmp_path / "out")
    assert info["status"] == "dropped" and "usable features" in info["reason"]


# ---------------------------------------------------------------- adapters

def test_myopia_adapter_reads_both_codings(tmp_path):
    for yes, no, fem in (("Yes", "No", "Female"), (2, 1, 2)):
        d = tmp_path / str(yes) / "myopia"
        d.mkdir(parents=True)
        pd.DataFrame({"id": [1, 2, 3], "studyyear": [1990] * 3, "myopic": [yes, no, yes], "age": [6, 7, 6],
                      "gender": [fem, "Male" if fem == "Female" else 1, fem], "spheq": [0.1, 0.5, -0.2],
                      "mommy": [yes, no, no], "dadmy": [no, no, yes]}).to_csv(d / "myopia.csv", index=False)
        r = RD.load_myopia(root=tmp_path / str(yes))
        assert r.y.tolist() == [1, 0, 1] and r.X.gender.tolist() == [1, 0, 1]
        assert "id" not in r.X and "studyyear" not in r.X and r.X.mommy.tolist() == [1, 0, 0]


def test_arff_reader(tmp_path):
    f = tmp_path / "x.arff"
    f.write_text("% comment\n@relation r\n@attribute '0' numeric\n@attribute '1' numeric\n@attribute Class {0,1}\n"
                 "@data\n1,0.5,1\n0,0.25,0\n")
    D = RD._read_arff(f)
    assert D.shape == (2, 3) and D.iloc[0, 1] == 0.5


def test_not_ready_is_clear(tmp_path):
    with pytest.raises(RD.NotReady, match="adapter for papila not finished"):
        RD.load("papila", root=tmp_path)


# ---------------------------------------------------------------- scoring logic

def test_verdict_order():
    base = dict(auc_S_Q=0.80, auc_classical_best=0.70, auc_nonlinear_best=0.70, auc_S_MF=0.70, auc_S_J0=0.6, auc_S_GS=0.6)
    assert RL.verdict(base) == "exact ahead (candidate)"
    assert RL.verdict({**base, "auc_nonlinear_best": 0.795}) == "classical enough"
    assert RL.verdict({**base, "auc_S_MF": 0.792}) == "mean-field enough"
    assert RL.verdict({**base, "auc_S_MF": 0.77}) == "exact slightly ahead"


def test_boot_diff_and_groups():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good, bad = y + 0.1 * rng.standard_normal(300), rng.standard_normal(300)
    obs, lo, hi = RL.boot_diff(y, np.repeat(np.arange(150), 2), good, [bad], n=200)
    assert obs > 0.3 and lo > 0.2 and hi < 0.7
    obs2, lo2, _ = RL.boot_diff(y, np.arange(300), bad, [good], n=200)
    assert obs2 < 0 and lo2 < obs2


def test_family_size():
    assert RL.family_size("XYZ|k2|tau5", 8) == 24
    assert RL.family_size("full|k1|tau3", 8) == 108
    assert RL.family_size("ZZ|k2", 8) == 28 and RL.family_size("Zfull|k2", 8) == 36
    assert len(RL.bank_specs()) == 12 + 2 * 5 * 12 + 5


def test_run_dataset_grouped_smoke():
    """the core student code accepts groups and keeps the ungrouped path unchanged"""
    import teacher_student as ts
    X, y, g = _fake_table(n=120, k=4)
    Z = X.to_numpy()
    S = {"S_Q": {"lin": Z}, "S_MF": {"lin": Z ** 2}}
    C = {("Poly", 4): Z}
    R1, _ = ts.run_dataset(Z, y, {"name": "t"}, S=S, C=C)
    R2, _ = ts.run_dataset(Z, y, {"name": "t"}, S=S, C=C, groups=g)
    assert set(R1.arm) == set(R2.arm) == {"S_Q", "S_MF", "classical_best"}
    assert R1.auc.notna().all() and R2.auc.notna().all()
