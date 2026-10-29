"""Tests for S1 Part B (studies/s1_scaling/run.py partB_*, studies/common/torchlearn.py).

The first group needs no PyTorch.  The torch group is skipped where PyTorch isn't installed
(e.g. a laptop) and runs on the GPU machine; the GPU test is skipped when no GPU is visible."""
import importlib.util
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("s1run_b", ROOT / "studies" / "s1_scaling" / "run.py")
S1 = importlib.util.module_from_spec(_spec)
sys.modules["s1run_b"] = S1
_spec.loader.exec_module(S1)


# ---------------------------------------------------------------------------
# no torch needed
# ---------------------------------------------------------------------------

def test_splits_are_nested_and_disjoint():
    prev_tr = None
    for n in S1.PARTB["n_grid"]:
        tr, va, te = S1.splits(n, 0.2)
        assert len(tr) + len(va) == n and len(va) == round(0.2 * n)
        assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
        assert te.min() == 30_000 and len(te) == 10_000
        if prev_tr is not None:
            assert set(prev_tr) <= set(tr)                       # more data = a superset
        prev_tr = tr


def test_basis_sizes_and_matrices():
    p = 4
    sizes = S1.basis_sizes(p, S1.PARTB)
    polys = [s for s in sizes if s[0] == "poly"]
    assert [s[2] for s in polys] == [comb(p + d, d) - 1 for d in range(1, S1.d_max(p) + 1)]
    assert max(s[2] for s in polys) <= S1.DESIGN["max_terms"]
    assert {s[2] for s in sizes if s[0] == "rff"} == {p * m for m in S1.PARTB["fourier_mults"]}
    Z = np.random.default_rng(0).standard_normal((50, p))
    for fam, _, D in [("poly", "d2", comb(p + 2, 2) - 1), ("fourier", "D8", 8), ("rff", "D16", 16)]:
        X = S1.basis_matrix(Z, fam, D, p)
        assert X.shape == (50, D) and np.isfinite(X).all()
    # degree-1 Hermite = the inputs themselves
    assert np.allclose(S1.basis_matrix(Z, "poly", p, p), Z)


def test_smoke_config_is_small():
    c = S1.partB_cfg(smoke=True)
    assert c["ps"] == [4] and max(c["n_grid"]) <= 1_000 and c["mlp"]["max_steps"] <= 300
    assert S1.partB_dir(True) != S1.partB_dir(False)


def _fake_P(rng):
    """a synthetic Part B table: 2 teachers, 2 n; family AUCs chosen so the answer is known"""
    rows = []
    for nm in ("t1", "t2"):
        for n in (300, 30_000):
            rows.append(dict(name=nm, learner="ref", config="own", size=12, n=n, val_loss=0.4,
                             val_auc=0.9, test_auc=0.90))
            # poly: two sizes; the one with lower val loss has test 0.70
            rows.append(dict(name=nm, learner="poly", config="d1", size=4, n=n, val_loss=0.60,
                             val_auc=0.7, test_auc=0.70))
            rows.append(dict(name=nm, learner="poly", config="d2", size=14, n=n, val_loss=0.65,
                             val_auc=0.8, test_auc=0.85))     # better test, worse val: must NOT be picked
            rows.append(dict(name=nm, learner="mlp", config="w16_d2", size=100, n=n, val_loss=0.45,
                             val_auc=0.88, test_auc=0.895 if n == 30_000 else 0.80))
            rows.append(dict(name=nm, learner="gbt", config="L8_R100", size=800, n=n, val_loss=0.50,
                             val_auc=0.85, test_auc=0.84))
    return pd.DataFrame(rows)


def test_selection_uses_validation_not_test():
    S = S1.selected_table(_fake_P(np.random.default_rng(0)))
    fx = S[(S.choice == "fixed") & (S.n == 300)]
    assert (fx.config == "d1").all() and np.allclose(fx.shortfall, 0.20)
    ln = S[(S.choice == "learned") & (S.n == 30_000)]
    assert (ln.learner == "mlp").all() and np.allclose(ln.shortfall, 0.005)
    ns = S1.n_star(S, "learned", 0.01)
    assert (ns.n_star == 30_000).all()
    assert np.allclose(S[S.choice == "ref"].shortfall if "ref" in set(S.choice) else 0, 0)


def test_checks_partB_on_a_known_table():
    """window cells: fixed short by 0.2, learned short by 0.005 at every p -> B1 holds, B2 fails
    (learned gets there in 12/12), B0 holds; slopes 0 for both kinds -> B3a fails, B3b holds"""
    rows = []
    cl = S1.cells()
    for p in S1.PARTB["ps"]:
        for kind in ("Q", "MF"):
            for c in cl:
                for s in range(3):
                    nm = f"x_{kind}_{p}_{c['c']}_{c['tau']}_{s}"
                    for ch, sf in (("fixed", 0.2), ("learned", 0.005)):
                        rows.append(dict(name=nm, n=30_000, choice=ch, shortfall=sf, p=p, kind=kind,
                                         c=c["c"], tau=c["tau"], seed=s, region=c["region"]))
    K = S1.checks_partB(pd.DataFrame(rows)).set_index("check")
    assert bool(K.loc["B0", "held"]) and bool(K.loc["B1", "held"]) and not bool(K.loc["B2", "held"])
    assert not bool(K.loc["B3a", "held"]) and bool(K.loc["B3b", "held"])


def test_gbt_job_end_to_end(tmp_path, monkeypatch):
    """a tiny tree job on a fake teacher file, through the real code path"""
    p = 4
    rng = np.random.default_rng(0)
    Z = S1.bs.pool_inputs(p)[:S1.n_rows()]
    Y = np.stack([(Z[:, 0] + 0.3 * rng.standard_normal(len(Z)) > 0), (Z[:, 1] > 0)], 1).astype(np.float32)
    monkeypatch.setattr(S1, "load_teachers", lambda q: (Y, ["a", "b"]))
    monkeypatch.setattr(S1, "partB_dir", lambda smoke=False: tmp_path)
    cfg = dict(S1.PARTB, gbt_rounds=[5, 20], n_grid=[300])
    monkeypatch.setattr(S1, "partB_cfg", lambda smoke=False: cfg)
    st, _ = S1.gbt_job((p, 300, 8, 0, 2, False))
    assert st.startswith("ok")
    G = pd.read_csv(next(tmp_path.glob("gbt_*.csv")))
    assert len(G) == 4 and set(G.rounds) == {5, 20} and set(G.name) == {"a", "b"}
    assert (G[G.name == "b"].test_auc > 0.9).all()           # y = [z_1 > 0] is easy for trees
    assert S1.gbt_job((p, 300, 8, 0, 2, False))[0] == "skip"   # restartable


def test_scoreB_end_to_end_on_fake_files(tmp_path, monkeypatch, capsys):
    """do_scoreB on synthetic Part B files for every teacher: runs, writes both tables"""
    rng = np.random.default_rng(0)
    out = tmp_path / "s1"
    (out / "partB").mkdir(parents=True)
    monkeypatch.setattr(S1, "OUT", out)
    monkeypatch.setattr(S1, "partB_dir", lambda smoke=False: out / "partB")
    M = pd.DataFrame([t for p in S1.PARTB["ps"] for t in S1.teacher_specs(p)])
    M.to_csv(out / "S1_TEACHERS.csv", index=False)
    rows = []
    for nm in M.name:
        for n in S1.PARTB["n_grid"]:
            rows.append(dict(name=nm, learner="ref", config="own_XYZ", size=12, n=n, val_loss=0.4,
                             val_auc=0.9, test_auc=0.9))
            for lr, cf in (("poly", "d1"), ("fourier", "D8"), ("rff", "D8"), ("mlp", "w16_d2"), ("gbt", "L8_R100")):
                a = 0.6 + 0.3 * rng.random()
                rows.append(dict(name=nm, learner=lr, config=cf, size=10, n=n, val_loss=1 - a,
                                 val_auc=a, test_auc=a))
    pd.DataFrame(rows).to_csv(out / "partB" / "fake.csv", index=False)
    S1.do_scoreB()
    assert (out / "S1_PARTB.csv").exists() and (out / "S1_PARTB_CHECKS.csv").exists()
    K = pd.read_csv(out / "S1_PARTB_CHECKS.csv")
    assert list(K.check) == ["B0", "B1", "B2", "B3a", "B3b"]
    assert "Part B checks" in capsys.readouterr().out


def test_scoreB_long_end_to_end_on_fake_files(tmp_path, monkeypatch, capsys):
    """long-trained networks that are better on validation (and test) for 2 of every 3 teachers
    must lower the 'learned_long' median shortfall; R1/R2 are computed on the p = 8 cells"""
    out = tmp_path / "s1"
    (out / "partB").mkdir(parents=True)
    (out / "partB_long").mkdir(parents=True)
    monkeypatch.setattr(S1, "OUT", out)
    monkeypatch.setattr(S1, "partB_dir", lambda smoke=False: out / "partB")
    monkeypatch.setattr(S1, "long_dir", lambda smoke=False: out / "partB_long")
    M = pd.DataFrame(S1.teacher_specs(8))
    M.to_csv(out / "S1_TEACHERS.csv", index=False)
    P, L = [], []
    for k, nm in enumerate(M.name):
        P.append(dict(name=nm, learner="ref", config="own_XYZ", size=24, n=30_000, val_loss=0.3, val_auc=0.95, test_auc=0.95))
        P.append(dict(name=nm, learner="mlp", config="w16_d2", size=10, n=30_000, val_loss=0.5, val_auc=0.85, test_auc=0.85))
        P.append(dict(name=nm, learner="gbt", config="L8_R100", size=800, n=30_000, val_loss=0.55, val_auc=0.8, test_auc=0.8))
        good = k % 3 != 0
        L.append(dict(name=nm, learner="mlp_long", config="w64_d3_lr0.001", size=99, n=30_000,
                      val_loss=0.35 if good else 0.6, val_auc=0.9, test_auc=0.945 if good else 0.7,
                      best_step=5000, steps=12_000))
    pd.DataFrame(P).to_csv(out / "partB" / "fake_p8_x_n30000.csv", index=False)
    pd.DataFrame(L).to_csv(out / "partB_long" / "fake.csv", index=False)
    S1.do_scoreB_long()
    K = pd.read_csv(out / "S1_PARTB_LONG_CHECKS.csv").set_index("check")
    assert list(K.index) == ["R1", "R2"]
    S = pd.read_csv(out / "S1_PARTB_LONG.csv")
    lb = S[S.choice == "learned"].shortfall.median()
    la = S[S.choice == "learned_long"].shortfall.median()
    assert np.isclose(lb, 0.10) and np.isclose(la, 0.005)
    assert not bool(K.loc["R1", "held"])                     # a drop of 0.095 > 0.02
    assert "robustness checks" in capsys.readouterr().out
