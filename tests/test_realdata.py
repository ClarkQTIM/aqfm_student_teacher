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
    X, y, g = _fake_table(k=1)                                  # p_min is 2 (Amendment 3)
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
    with pytest.raises(RD.NotReady, match="adapter for keratoconus not finished"):
        RD.load("keratoconus", root=tmp_path)
    with pytest.raises(RD.NotReady, match="missing .*patient_data_od.xlsx"):
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


# ---------------------------------------------------------------- adapters on stand-ins in the real files' layouts

def _papila_sheet(path, rows):
    top = [None, "Age", "Gender", "Diagnosis", "Refractive_Defect", None, None, "Phakic/Pseudophakic", "IOP", None,
           "Pachymetry", "Axial_Length", "VF_MD"]
    sub = [None, "Age", "Gender", "Diagnosis", "dioptre_1", "dioptre_2", "astigmatism", "Phakic/Pseudophakic",
           "Pneumatic", "Perkins", "Pachymetry", "Axial_Length", "VF_MD"]
    idr = ["ID"] + [None] * 12
    pd.DataFrame([top, sub, idr] + rows).to_excel(path, header=False, index=False)


def test_papila_adapter(tmp_path, monkeypatch):
    d = tmp_path / "papila"
    d.mkdir()
    od = [["#001", 50, 0, 0, 0.5, -1, 90, 0, 18, None, 550, 23.5, None],
          ["#002", 60, 1, 1, 1.0, -0.5, 80, 0, 25, None, 530, 24.0, -5.2],
          ["#003", 55, 0, 2, 0.0, -0.25, 10, 1, 21, None, 560, 23.0, None]]
    os_ = [["#001", 50, 0, 0, 0.25, -1, 95, 0, 17, None, 552, 23.4, None],
           ["#002", 60, 1, 0, 0.75, -0.5, 85, 0, 19, None, 535, 23.9, None],
           ["#003", 55, 0, 1, 0.0, -0.25, 15, 1, 28, None, 558, 23.1, -3.0]]
    _papila_sheet(d / "patient_data_od.xlsx", od)
    _papila_sheet(d / "patient_data_os.xlsx", os_)
    monkeypatch.setattr(RD, "PAPILA_COUNTS", {0: 3, 1: 2, 2: 1})
    r = RD.load_papila(root=tmp_path)
    assert len(r.y) == 5 and r.y.sum() == 2                     # suspect eye dropped
    assert "vf_md" not in r.X and "id" not in r.X and "diagnosis" not in r.X
    assert {"refr_M", "refr_J0", "refr_J45"} <= set(r.X) and "astigmatism" not in r.X
    assert sorted(set(r.groups)) == ["#001", "#002", "#003"]
    monkeypatch.setattr(RD, "PAPILA_COUNTS", {0: 333, 1: 87, 2: 68})
    with pytest.raises(RD.NotReady, match="differ from the paper"):
        RD.load_papila(root=tmp_path)


def test_grape_adapter(tmp_path, monkeypatch):
    d = tmp_path / "grape"
    d.mkdir()
    top = ["Subject Number", "Laterality", "Age", "Gender", "IOP", "CCT", "Total Visits", "Progression Status", None, None,
           "Category of Glaucoma", "OCT RNFL thickness", None, None, None, None, "Corresponding CFP", "Acquisition Device",
           "Resolusion", "VF"] + [None] * 60
    sub = [None] * 7 + ["PLR2", "PLR3", "MD", None, "Mean", "S", "N", "I", "T", None, None, None] + list(range(61))
    rows = []
    for subj, eye, prog in ((1, "OD", 1), (None, "OS", 0), (2, "OD", 0)):
        rows.append([subj, eye, 60, "M", 18, 540, 5, prog, 0, 0, "OAG", 90, 110, 70, 115, 60, "x.jpg", "dev", "res"]
                    + [-1] + [20] * 60)
    with pd.ExcelWriter(d / "VF and clinical information.xlsx") as w:
        pd.DataFrame([top, sub] + rows).to_excel(w, sheet_name="Baseline", header=False, index=False)
    monkeypatch.setattr(RD, "GRAPE_EYES", 3)
    monkeypatch.setattr(RD, "GRAPE_PLR2", 1)
    r = RD.load_grape(root=tmp_path)
    assert r.y.tolist() == [1, 0, 0] and list(r.groups) == ["1", "1", "2"]       # merged cell forward-filled
    assert "vf_60" in r.X and r.X.vf_0.isna().all()                               # -1 -> missing
    assert not {"Total Visits", "PLR3", "MD"} & set(r.X.columns)


def test_fives_adapter(tmp_path):
    d = tmp_path / "fives_vascx"
    d.mkdir()
    pd.DataFrame({"vascx_id": ["a", "b", "c"], "image_id": ["1_N", "2_D", "3_A"], "disease_code": ["N", "D", "A"],
                  "disease": ["Normal", "DR", "AMD"], "split": ["train", "test", "train"],
                  "cre_veins": [15.0, 16.0, np.nan], "sparsity": [0.1, 0.2, 0.3]}).to_csv(d / "fives_vascx_dataset.csv", index=False)
    r = RD.load_fives_vascx(root=tmp_path)
    assert r.y.tolist() == [0, 1, 1] and list(r.X.columns) == ["cre_veins", "sparsity"] and r.groups is None


def test_power_vectors():
    pv = RD.power_vectors(pd.Series([0.75, -2.0]), pd.Series([-1.75, -1.0]), pd.Series([90, 180]))
    assert np.allclose(pv.refr_M, [-0.125, -2.5])
    assert np.allclose(pv.refr_J0, [-0.875, 0.5]) and np.allclose(pv.refr_J45, [0, 0], atol=1e-12)
    same = RD.power_vectors(pd.Series([0.0, 0.0]), pd.Series([-1.0, -1.0]), pd.Series([0, 180]))
    assert np.allclose(same.iloc[0], same.iloc[1])                  # 0 and 180 degrees are the same astigmatism


def test_missing_datasets_are_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(RD, "DATA_ROOT", tmp_path / "data")
    RL.do_prepare(tmp_path / "out", ["keratoconus"])
    info = json.loads((tmp_path / "out" / "keratoconus" / "PREP.json").read_text())
    assert info["status"] == "not obtained" and "not finished" in info["reason"]
    row = RL.score_dataset(tmp_path / "out", "keratoconus")
    assert row["status"] == "not obtained" and row["reason"]


# ---------------------------------------------------------------- the Run-2 pool (Amendment 3)

def test_pool_list_matches_run2():
    assert len(RD.POOL) == 55 and len(set(RD.POOL)) == 55
    assert set(RD.POOL_BOOLEAN) <= set(RD.POOL) and len(RD.POOL_BOOLEAN) == 9
    assert RL.DESIGN["pool"] == RD.POOL and RL.DESIGN["p_min"] == 2


def test_binarize_rule():
    y, info = RD.binarize([0, 1, 1, 0], "x")
    assert y.tolist() == [0, 1, 1, 0] and info["classes"] == 2
    y, _ = RD.binarize(["absent", "present", "present"], "x")
    assert y.tolist() == [0, 1, 1]
    y, info = RD.binarize([3, 1, 1, 2, 1, 3], "x")                       # largest class (1) vs rest
    assert y.tolist() == [0, 1, 1, 0, 1, 0] and info["positive_class"] == "1"
    y, info = RD.binarize([0, 1, 2, 0, 1, 2], "x")                       # tie: first in sorted order
    assert info["positive_class"] == "0"


def _fake_pmlb(root, name, n, p, seed=0, classes=2):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    D["target"] = rng.integers(0, classes, n)
    d = root / "pmlb_cache" / name
    d.mkdir(parents=True)
    D.to_csv(d / f"{name}.tsv.gz", sep="\t", index=False)


def test_pool_loader_reads_cache_and_subsamples(tmp_path):
    _fake_pmlb(tmp_path, "magic", 5000, 10)
    _fake_pmlb(tmp_path, "yeast", 400, 8, classes=5)
    r = RD.load("pmlb_magic", root=tmp_path / "ophtho")
    assert len(r.y) == 3000 and r.info["subsampled_from"] == 5000 and r.X.shape[1] == 10
    assert abs(r.y.mean() - 0.5) < 0.05
    r2 = RD.load("pmlb_magic", root=tmp_path / "ophtho")
    assert r2.X.equals(r.X)                                              # seeded: the same subsample every time
    r = RD.load("pmlb_yeast", root=tmp_path / "ophtho")
    assert len(r.y) == 400 and r.info["classes"] == 5 and r.info["subsampled_from"] is None
    with pytest.raises(RD.NotReady, match="pmlb_cache"):
        RD.load("pmlb_monk2", root=tmp_path / "ophtho")


def test_pool_bundled_legacy_sets():
    r = RD.load("iris")
    assert len(r.y) == 150 and r.X.shape[1] == 4 and r.y.sum() == 50   # setosa (tie -> first) vs rest
    r = RD.load("wine")
    assert len(r.y) == 178 and r.y.sum() == 71                          # class 1 (largest) vs rest


def test_pool_checks():
    T = pd.DataFrame({"name": ["pmlb_monk2", "a", "b"], "verdict": ["exact ahead (candidate)", "classical enough",
                      "classical enough"], "confirmed_win": [True, False, False], "confirmed_compact": [False] * 3,
                      "confirmed_quantum_part": [False] * 3, "auc_S_Q": [0.8, 0.7, 0.7], "auc_S_J0": [0.7, 0.7, 0.695],
                      "F1_order": ["3", "1", "1"], "margin_disc": [0.06, -0.01, 0.0],
                      "boolean_rule": [True, False, False]})
    K = RL.checks_pool(T).set_index("check")
    assert K.loc["Q2", "count"] == 1 and "pmlb_monk2" in K.loc["Q2", "prediction"]
    assert K.loc["Q5", "count"] == 2
