"""Tests for the geometric-difference analysis (experiments/ts_geometry.py).

1. The math: our regularized g reduces to Huang et al.'s Eq. 5 as lambda -> 0, a kernel
   compared with itself gives g <= 1, and the two bounds the plan relies on
   (s_C <= g^2 s_T, training error <= t s_T) hold for labels the target can express.
2. A tiny end-to-end run of one tau-sweep cell and one real dataset (p small, few settings).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ts_geometry as G  # noqa: E402
import ts_tau_sweep as tt  # noqa: E402
from aqfm_prescreen.aqfm import compute_bank  # noqa: E402


def _rng(s=0):
    return np.random.default_rng(s)


# ---------------------------------------------------------------- math

def test_regularized_g_tends_to_eq5():
    N = 150
    X, Y = _rng(0).standard_normal((N, 4)), _rng(1).standard_normal((N, 4))
    KT = G.trace_normalize(G.rbf_kernels(X)[1.0] + 1e-3 * np.eye(N))
    KC = G.trace_normalize(G.rbf_kernels(Y, [0.05])[0.05] + 1e-1 * np.eye(N))   # well conditioned
    g5 = G.geometric_difference_eq5(KT, KC)
    g, t = G.geometric_difference(G.Eig(KT), G.Eig(KC), 1e-8)
    assert g == pytest.approx(g5, rel=1e-4)
    assert t < 1e-10


def test_g_of_kernel_with_itself_is_at_most_one():
    K = G.linear_kernel(_rng(2).standard_normal((120, 6)))
    for lam in G.GEO["lambdas"]:
        g, _ = G.geometric_difference(G.Eig(K), G.Eig(K), lam)
        assert g <= 1 + 1e-9


def test_bounds_hold_for_labels_the_target_can_express():
    rng = _rng(3)
    N = 120
    eT = G.Eig(G.linear_kernel(rng.standard_normal((N, 8))))
    eC = G.Eig(G.rbf_kernels(rng.standard_normal((N, 3)))[1.0])
    S = (eT.V * np.sqrt(eT.e)) @ eT.V.T                               # sqrt(K_T)
    r = eT.e > 0
    for lam in G.GEO["lambdas"]:
        g, t = G.geometric_difference(eT, eC, lam)
        for _ in range(50):
            v = eT.V[:, r] @ rng.standard_normal(r.sum())             # v in the target's range
            y = S @ v                                                  # labels with s_T = |v|^2
            s, tr = G.complexity(eC, y, lam)
            assert s <= g ** 2 * (v @ v) * (1 + 1e-9)
            assert tr <= t * (v @ v) * (1 + 1e-9)


def test_kernels_are_trace_normalized_and_constant_columns_dropped():
    F = np.c_[_rng(4).standard_normal((50, 3)), np.ones(50)]
    K = G.linear_kernel(F)
    assert np.trace(K) == pytest.approx(50)
    assert np.linalg.matrix_rank(K) == 3
    assert G.linear_kernel(np.ones((10, 2))) is None
    for Kr in G.rbf_kernels(F).values():
        assert np.allclose(np.diag(Kr), 1.0)


# ---------------------------------------------------------------- end to end, tau sweep

def test_tt_cell_job_runs_on_a_tiny_cell(tmp_path):
    p, kap, tau = 4, 2.0, 3.0
    (tmp_path / "banks" / f"p{p}").mkdir(parents=True)
    for spec in [(p, "Q", kap, tau), (p, "MF", kap, tau), (p, "J0", 0.0, tau), (p, "GS", kap, 0.0)]:  # GS: make_labels needs it
        tt._bank_job((tmp_path,) + spec)
    (tmp_path / "labels").mkdir()
    rows = []
    for kind in ("Q", "MF"):
        for ro in ("local", "zlocal"):
            sp = dict(kind=kind, p=p, kappa=kap, tau=tau, readout=ro, seed=0)
            sp["name"] = f"tt_{kind}_p{p}_k{kap:g}_t{tau:g}_{ro}_s0"
            sp["data_seed"] = tt._seed(sp["name"])
            y, meta = tt.make_labels(tmp_path, sp)
            np.save(tmp_path / "labels" / f"{sp['name']}.npy", y)
            rows.append(dict(name=sp["name"], kind=kind, readout=ro))
    teachers = pd.DataFrame(rows)
    name, status = G.tt_cell_job((tmp_path, p, kap, tau, teachers))
    assert status.startswith("ok")
    R = pd.read_csv(tmp_path / "geometry" / "tt" / f"{name}.csv")
    # 4 targets x (3 classical + 3+1 for each of 2 twins) emulators x 4 lambdas
    assert len(R) == 4 * (3 + 2 * 4) * len(G.GEO["lambdas"])
    assert np.isfinite(R.g).all() and (R.g >= 0).all() and (R.t >= 0).all()
    assert set(R.emulator) == {"classical", "MF_rbf", "MF_lin", "J0_rbf", "J0_lin", "Q_rbf", "Q_lin"}
    L = pd.read_csv(tmp_path / "geometry" / "tt" / f"{name}_labels.csv")
    assert set(L.name) == set(teachers.name) and (L.s >= 0).all()
    assert G.tt_cell_job((tmp_path, p, kap, tau, teachers))[1] == "skip"      # restartable


# ---------------------------------------------------------------- end to end, real data

def test_real_job_runs_on_a_tiny_dataset(tmp_path, monkeypatch):
    monkeypatch.setitem(tt.DESIGN, "student_kappas", [1.0, 2.0])
    monkeypatch.setitem(tt.DESIGN, "student_taus", [1.0, 3.0])
    o = tmp_path / "toy"
    (o / "banks").mkdir(parents=True)
    rng = _rng(5)
    Z = rng.standard_normal((40, 2))
    y = (Z[:, 0] + 0.5 * rng.standard_normal(40) > 0).astype(int)
    np.savez(o / "prepared.npz", Z=Z, y=y)
    for kind, kp, ta in [("J0", 0.0, 1.0), ("J0", 0.0, 3.0)] + \
            [(k, kp, ta) for k in ("Q", "MF") for kp in (1.0, 2.0) for ta in (1.0, 3.0)]:
        J = 0.0 if kind == "J0" else kp / 1
        B = compute_bank(Z, J, ta, method="meanfield" if kind == "MF" else "strang")
        np.save(o / "banks" / f"{tt.bank_key(kind, kp, ta)}.npy", np.asarray(B, float))
    pd.DataFrame(dict(arm=["S_Q"] * 3, choice=["XYZ|k1|tau3"] * 3)).to_csv(o / "discovery_students.csv",
                                                                           index=False)
    (o / "PREP.json").write_text(json.dumps({"status": "ok"}))
    name, status = G.real_job((tmp_path, "toy"))
    assert status.startswith("ok")
    R = pd.read_csv(o / "geometry.csv")
    assert set(R.setting) == {"grid", "chosen"}
    assert (R[R.setting == "grid"].groupby(["kappa", "tau"]).size() > 0).sum() == 4
    assert R[R.setting == "chosen"].family.iloc[0] == "XYZ"
    assert R.n_used.iloc[0] == 40
    assert G._real_names(tmp_path) == ["toy"]


def test_real_rows_subsample_is_seeded_and_capped():
    a = G._real_rows(5000, "x", 600)
    assert len(a) == 600 and len(set(a)) == 600 and np.array_equal(a, G._real_rows(5000, "x", 600))
    assert np.array_equal(G._real_rows(300, "x", 600), np.arange(300))


def test_workers_capped():
    import subprocess
    r = subprocess.run([sys.executable, str(Path(G.__file__)), "score", "--workers", "40"],
                       capture_output=True, text=True)
    assert "STOP" in (r.stdout + r.stderr)


def test_score_tt_and_checks_run_on_fake_tables(tmp_path):
    """score plumbing only: fake geometry rows for two cells + a fake ADDONS table"""
    rows = []
    for tau, region, gscale in ((5.0, "window", 10.0), (1.0, "below (weak coupling or short anneal)", 1.0)):
        for kind in ("Q", "MF"):
            for ro in ("local", "zlocal"):
                for em in ("classical", "MF_rbf", "Q_rbf", "J0_rbf", "MF_lin"):
                    for c in (0.25, 1.0, 4.0):
                        for lam in G.GEO["lambdas"]:
                            rows.append(dict(kind=kind, p=4, kappa=2.0, tau=tau, readout=ro, emulator=em,
                                             c=c, lam=lam, g=gscale * c * (0.5 if ro == "zlocal" else 1),
                                             t=0.01, rank_T=12))
    d = tmp_path / "geometry" / "tt"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "p4_k2_t5.csv", index=False)
    T = []
    for tau, region in ((5.0, "window"), (1.0, "below (weak coupling or short anneal)")):
        for kind in ("Q", "MF"):
            for ro in ("local", "zlocal"):
                for s in range(3):
                    hard = tau == 5.0 and ro == "local"
                    T.append(dict(name=f"{kind}{tau}{ro}{s}", kind=kind, p=4, kappa=2.0, tau=tau, readout=ro,
                                  region=region, best_student_auc=0.9, auc_classical_best=0.8 if hard else 0.9,
                                  auc_nonlinear_best=0.75, gap_Q_MF=0.1 if hard else 0.0))
    pd.DataFrame(T).to_csv(tmp_path / "ADDONS_DATASETS.csv", index=False)
    _, TT = G.score_tt(tmp_path, G.GEO["lam_primary"])
    assert TT.g_classical.notna().all()
    assert (TT.g_classical == TT.groupby(["tau", "readout"]).g_classical.transform("first")).all()
    K = G.checks_tt(TT, G.GEO["lam_primary"])
    assert set(K.check) == {"G1", "G2", "G3", "G4", "G5"}
    assert bool(K[K.what.str.startswith("classical: AUROC")].held.iloc[0])
