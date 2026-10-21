#!/usr/bin/env python3
"""
Teacher-student study: when does ENTANGLEMENT in the map matter, and how many
classical features does it take to catch up?

Teachers (how labels are made, from globally standardized Gaussian inputs):
    J0   no couplings (product state, exact)
    MF   uniform coupling J = kappa/(p-1), mean-field (interactions, no entanglement)
    Q    same coupling, exact dynamics (interactions + entanglement)
  readout of the teacher's observables:
    local    random weights on <X_i>, <Y_i>, <Z_i>
    pairs    random weights on <X_iX_j>, <Y_iY_j>, <Z_iZ_j>
    collect  sum_i <X_i>   (the dial's "collective" target)
  label  y ~ Bernoulli(sigmoid(score / TEMP)), score standardized (soft labels)

Students (each tuned by inner CV over its own observables, J, tau and the LR's C):
    S_J0, S_MF, S_Q
    classical_best   inner-CV choice over every classical map below + the v4 suite
  classical cost curves: Fourier, random Fourier features, polynomial, splines
    at growing feature counts D (C tuned on inner folds at every point)  ->  D* = smallest D whose mean AUC reaches
    S_Q's mean AUC - 0.01 (descriptive; "unreachable" if none within the grid)

Per teacher we also record:
    entanglement  E = mean over samples and qubits of 1 - |r_i|^2, with
                  r_i = (<X_i>,<Y_i>,<Z_i>)  (0 for product states)
    mf_agreement  corr(teacher score, same readout applied to the mean-field twin)
    oracle_auc    AUC of the true score on the drawn labels (the ceiling)

"Entanglement matters"  <=>  S_Q beats S_MF.   "Interactions matter"  <=>  S_MF beats S_J0.

Inputs are standardized once per dataset (label-free) and shared by every arm.
Outer CV 5 folds, inner 3, as in the dial.

Phases:
  make   write datasets + TS_TEACHERS.csv + TS_PROTOCOL.json (hypotheses, written BEFORE any student runs)
  run    one job per dataset -> OUT/runs/<name>.csv (+ _curve.csv); restartable
  score  TS_DATASETS.csv, TS_SUMMARY.csv, hypothesis checks
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from aqfm_prescreen.aqfm import AQFM_FAMILIES, compute_bank, family_matrix, slices_for_p
from aqfm_prescreen.dial import _global_z, classical_candidates, lr_auc
from aqfm_prescreen.dictionaries import balanced_fourier

# ---------------------------------------------------------------------------
# design (pilot defaults; all recorded in TS_PROTOCOL.json)
# ---------------------------------------------------------------------------

DESIGN = {
    "n": 600,
    "ps": [4, 6, 8],
    "kappas": [0.5, 1.0, 2.0, 4.0, 8.0],
    "teacher_taus": [1.0, 5.0],
    "readouts": ["local", "pairs", "collect"],
    "seeds_J0": 2,              # J0 teachers have no kappa -> 2 seeds instead
    "seeds_coupled": 1,
    "temp": 0.25,
    # student grids CONTAIN every teacher setting (well-specified students)
    "student_kappas": [0.5, 1.0, 2.0, 4.0, 8.0],
    "student_taus": [0.3, 1.0, 3.0, 5.0, 10.0],
    "curve_mults": [1, 2, 4, 8, 16, 32, 64, 128],   # D = mult * p
    "poly_degrees": [1, 2, 3, 4, 5, 6],
    "spline_knots": [3, 5, 9, 17, 33, 65],
    "outer": 5, "inner": 3, "cv_seed": 42,
    "dstar_tol": 0.01,
    # v2 (amendment, see AMENDMENT): L2 strength tuned on the inner folds for every arm
    "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "version": 2,
}

AMENDMENT = {
    "version": 2,
    "why": ("v1 results (2026-10-13) showed a broken classical arm: (a) label-tuned suite maps "
            "were fit on all outer-train rows and scored on inner folds of those same rows, "
            "inflating their inner AUC by ~0.07, so classical_best always picked GA2MPairs and "
            "tested below plain splines/Fourier; (b) LR used C=1 for every map, so large "
            "classical maps overfit and D* was 'unreachable' even for J0 teachers."),
    "changes": ["suite maps refit inside every inner fold",
                "C chosen on the inner folds from C_grid for every arm (students, curves, classical_best)",
                "curve rows also keep auc_C1 (the v1 setting) for comparison"],
    "unchanged": "datasets, teachers, student grids, CV splits, hypotheses H1-H6/R1",
    "note": "amendment made AFTER seeing v1 results; student-vs-student hypotheses were already scored in v1",
}

HYPOTHESES = {
    "H1_diagonal": "each teacher kind is learned best (within 0.01 AUC) by its own student kind (J0->S_J0, MF->S_MF, Q->S_Q) on >= 70% of datasets",
    "H2_interactions": "MF teachers with kappa >= 1: S_MF - S_J0 >= 0.01 on >= 50%",
    "H2b_weak_coupling": "Q teachers with kappa <= 1: |S_Q - S_MF| < 0.01 on >= 50% (mean-field suffices at weak coupling)",
    "H3_entanglement": "Q teachers: Spearman(teacher entanglement E, S_Q - S_MF) > 0.3",
    "H4_window": "Q teachers: median S_Q - S_MF is largest at kappa in {2, 4} and smaller at kappa = 8",
    "H5_cost": "D* = classical features needed to reach the BEST student's AUC - 0.01: median ordering J0 < MF < Q, and the fraction 'unreachable' is highest for Q teachers",
    "R1_report": "report the fraction of Q teachers with S_Q - S_MF < 0.01 ('made with J=1, does not need it')",
    "H6_mf_agreement": "Q teachers: Spearman(mf_agreement, S_Q - S_MF) < -0.3 (when the mean-field twin already reproduces the teacher's score, entanglement adds nothing)",
}


def teacher_specs(d=DESIGN):
    specs = []
    for p, tau, ro in product(d["ps"], d["teacher_taus"], d["readouts"]):
        for s in range(d["seeds_J0"]):
            specs.append(dict(kind="J0", p=p, kappa=0.0, tau=tau, readout=ro, seed=s))
        for kind, kap in product(("MF", "Q"), d["kappas"]):
            for s in range(d["seeds_coupled"]):
                specs.append(dict(kind=kind, p=p, kappa=kap, tau=tau, readout=ro, seed=s))
    for sp in specs:
        sp["name"] = (f"ts_{sp['kind']}_p{sp['p']}_k{sp['kappa']:g}_t{sp['tau']:g}_"
                      f"{sp['readout']}_s{sp['seed']}")
        sp["data_seed"] = int(hashlib.sha256(sp["name"].encode()).hexdigest()[:8], 16)
    return specs


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------

def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def _readout_matrix(B, readout, p):
    if readout == "local":
        return family_matrix(B, "XYZ", p)
    if readout == "pairs":
        return family_matrix(B, "pairs", p)
    if readout == "collect":
        return family_matrix(B, "X", p).sum(1, keepdims=True)
    raise ValueError(readout)


def entanglement(B, p):
    """mean over samples and qubits of 1 - |r_i|^2 (single-qubit linear entropy x2)"""
    sl, _ = slices_for_p(p)
    r2 = B[:, sl["X"]] ** 2 + B[:, sl["Y"]] ** 2 + B[:, sl["Z"]] ** 2
    return float(np.mean(1.0 - r2))


def make_dataset(sp, d=DESIGN):
    rng = np.random.default_rng(sp["data_seed"])
    p, n = sp["p"], d["n"]
    X = rng.standard_normal((n, p))
    Zg = _global_z(X)
    J = 0.0 if sp["kind"] == "J0" else sp["kappa"] / max(p - 1, 1)
    method = "meanfield" if sp["kind"] == "MF" else "strang"
    B = compute_bank(Zg, J, sp["tau"], method=method).astype(float)
    F = _readout_matrix(B, sp["readout"], p)
    mu, sd = F.mean(0), F.std(0) + 1e-12
    w = np.ones(F.shape[1]) if sp["readout"] == "collect" else rng.standard_normal(F.shape[1])
    score = _std(((F - mu) / sd) @ w)
    prob = 1.0 / (1.0 + np.exp(-score / d["temp"]))
    y = (rng.random(n) < prob).astype(int)
    meta = dict(sp, J=J, entanglement=entanglement(B, p), oracle_auc=roc_auc_score(y, score),
                pos_rate=float(y.mean()))
    if sp["kind"] == "Q":            # how well does the mean-field twin reproduce this teacher?
        Bm = compute_bank(Zg, J, sp["tau"], method="meanfield").astype(float)
        Fm = _readout_matrix(Bm, sp["readout"], p)
        sm = _std(((Fm - Fm.mean(0)) / (Fm.std(0) + 1e-12)) @ w)
        meta["mf_agreement"] = float(np.corrcoef(score, sm)[0, 1])
        meta["mf_feature_residual"] = float(np.mean(np.abs(F - Fm)))
    else:
        meta["mf_agreement"] = 1.0 if sp["kind"] == "MF" else np.nan
        meta["mf_feature_residual"] = 0.0 if sp["kind"] == "MF" else np.nan
    return X, y, meta


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------

def student_sets(Z, d=DESIGN):
    """{arm: {config: F}} for the three AQFM students (features for ALL rows)."""
    p = Z.shape[1]
    S = {"S_J0": {}, "S_MF": {}, "S_Q": {}}
    for tau in d["student_taus"]:
        B0 = compute_bank(Z, 0.0, tau, method="strang").astype(float)
        for fam in AQFM_FAMILIES:
            S["S_J0"][f"{fam}|tau{tau:g}"] = family_matrix(B0, fam, p)
        for kap in d["student_kappas"]:
            J = kap / max(p - 1, 1)
            for arm, method in (("S_MF", "meanfield"), ("S_Q", "strang")):
                B = compute_bank(Z, J, tau, method=method).astype(float)
                for fam in AQFM_FAMILIES:
                    S[arm][f"{fam}|k{kap:g}|tau{tau:g}"] = family_matrix(B, fam, p)
    return S


def curve_sets(Z, d=DESIGN):
    """{(family, D): F} classical maps at growing D (label-free, fit on all rows)."""
    p = Z.shape[1]
    C = {}
    for m in d["curve_mults"]:
        D = m * p
        C[("Fourier", D)] = balanced_fourier(Z, D)
        C[("RFF", D)] = RBFSampler(gamma=1.0 / p, n_components=D, random_state=0).fit(Z).transform(Z)
    for deg in d["poly_degrees"]:
        D = comb(p + deg, deg) - 1
        if D <= max(d["curve_mults"]) * p:
            C[("Poly", D)] = PolynomialFeatures(deg, include_bias=False).fit_transform(Z)
    for k in d["spline_knots"]:
        st = SplineTransformer(n_knots=k, degree=3, include_bias=False).fit(Z)
        F = st.transform(Z)
        C[("Spline", F.shape[1])] = F
    return C


def lr_path(Ftr, Fte, ytr, yte, Cs=None):
    """Binary LR test AUC for every C in Cs (ascending, warm-started).
    v2: every arm tunes the L2 strength; v1 used C = 1 everywhere, which let
    large classical maps overfit and made 'more features' look worse."""
    Cs = sorted(Cs or DESIGN["C_grid"])
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr), sc.transform(Fte)
    m = LogisticRegression(C=Cs[0], max_iter=3000, warm_start=True)
    out = np.empty(len(Cs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c in enumerate(Cs):
            m.set_params(C=c).fit(A, ytr)
            out[i] = roc_auc_score(yte, m.decision_function(B))
    return out


def _cv_path(pairs, yin, inner):
    """mean inner-fold AUC per C.  pairs[i] = (F_a, F_b) for inner fold i."""
    return np.mean([lr_path(Fa, Fb, yin[a], yin[b]) for (Fa, Fb), (a, b) in zip(pairs, inner)], axis=0)


def _pick(paths):
    """paths {key: inner AUC per C} -> (key, C, inner score) maximizing the inner score"""
    Cs = sorted(DESIGN["C_grid"])
    k = max(paths, key=lambda k: paths[k].max())
    i = int(np.argmax(paths[k]))
    return k, Cs[i], float(paths[k][i])


def _outer(Ftr, Fte, ytr, yte, C):
    Cs = sorted(DESIGN["C_grid"])
    return float(lr_path(Ftr, Fte, ytr, yte)[Cs.index(C)])


def run_dataset(X, y, meta, d=DESIGN, S=None, C=None, groups=None):
    """S, C: precomputed student / curve sets (the tau sweep shares them across
    datasets with the same inputs); computed here when not given.
    groups: optional patient ids (real data with two eyes per patient).  When given,
    outer AND inner folds keep each group in one fold (StratifiedGroupKFold); when
    None (all synthetic runs) the folds are exactly the original StratifiedKFold."""
    Z = _global_z(X)
    p = Z.shape[1]
    S = student_sets(Z, d) if S is None else S
    C = curve_sets(Z, d) if C is None else C
    Cs = sorted(d["C_grid"])
    if groups is None:
        outer = list(StratifiedKFold(d["outer"], shuffle=True, random_state=d["cv_seed"]).split(Z, y))
    else:
        groups = np.asarray(groups)
        outer = list(StratifiedGroupKFold(d["outer"], shuffle=True, random_state=d["cv_seed"]).split(Z, y, groups))
    rows, crows = [], []
    for f, (tr, te) in enumerate(outer):
        ytr, yte = y[tr], y[te]
        if groups is None:
            inner = list(StratifiedKFold(d["inner"], shuffle=True, random_state=0).split(tr, ytr))
        else:
            inner = list(StratifiedGroupKFold(d["inner"], shuffle=True, random_state=0).split(tr, ytr, groups[tr]))
        # AQFM students: joint inner-CV choice of (observables, J, tau, C)
        for arm, sets in S.items():
            paths = {k: _cv_path([(F[tr][a], F[tr][b]) for a, b in inner], ytr, inner)
                     for k, F in sets.items()}
            k, c, s_in = _pick(paths)
            F = sets[k]
            rows.append(dict(name=meta["name"], fold=f, arm=arm, choice=k, C=c, D=F.shape[1],
                             inner=s_in, auc=_outer(F[tr], F[te], ytr, yte, c)))
        # classical cost curve: every (family, D), C tuned on the inner folds
        cpaths = {}
        for (fam, D), F in C.items():
            cpaths[("curve", (fam, D))] = pth = _cv_path([(F[tr][a], F[tr][b]) for a, b in inner], ytr, inner)
            c = Cs[int(np.argmax(pth))]
            outp = lr_path(F[tr], F[te], ytr, yte)
            crows.append(dict(name=meta["name"], fold=f, family=fam, D=D, C=c,
                              auc=float(outp[Cs.index(c)]), auc_C1=float(outp[Cs.index(1.0)])))
        # classical_best: inner-CV choice over the curve maps + the v4 suite.
        # v2: label-tuned suite maps are refit INSIDE each inner fold (v1 fit them on all
        # outer-train rows, which inflated their inner scores by ~0.07 and made the
        # selector pick them, lowering classical_best)
        suite_out = {}
        for nm, D, fn in classical_candidates(p, "v2sym"):
            try:
                pairs = [fn(Z[tr][a], ytr[a], Z[tr][b]) for a, b in inner]
                o = fn(Z[tr], ytr, Z[te])
            except Exception:
                continue
            if o is None or any(q is None for q in pairs):
                continue
            arrs = [o[0], o[1]] + [x for q in pairs for x in q]
            if not all(np.isfinite(x).all() and np.abs(x).max() < 1e10 for x in arrs):
                continue
            cpaths[("suite", nm)] = _cv_path(pairs, ytr, inner)
            suite_out[nm] = o
        (kind, key), c, bs = _pick(cpaths)
        if kind == "curve":
            F = C[key]
            Ftr, Fte, choice = F[tr], F[te], f"{key[0]}{key[1]}"
        else:
            (Ftr, Fte), choice = suite_out[key], key
        rows.append(dict(name=meta["name"], fold=f, arm="classical_best", choice=choice, C=c,
                         D=Ftr.shape[1], inner=bs, auc=_outer(Ftr, Fte, ytr, yte, c)))
    return pd.DataFrame(rows), pd.DataFrame(crows)


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def do_make(out: Path, d=DESIGN):
    (out / "data").mkdir(parents=True, exist_ok=True)
    proto = {"design": d, "hypotheses": HYPOTHESES,
             "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "note": "written by `make`, before any student was run"}
    (out / "TS_PROTOCOL.json").write_text(json.dumps(proto, indent=2))
    metas = []
    for i, sp in enumerate(teacher_specs(d)):
        f = out / "data" / f"{sp['name']}.npz"
        X, y, meta = make_dataset(sp, d)
        np.savez_compressed(f, X=X, y=y)
        metas.append(meta)
        if i % 20 == 0:
            print(f"[make {i + 1}] {sp['name']}  E={meta['entanglement']:.3f}  "
                  f"oracle={meta['oracle_auc']:.3f}", flush=True)
    M = pd.DataFrame(metas)
    M.to_csv(out / "TS_TEACHERS.csv", index=False)
    print(f"made {len(M)} datasets -> {out/'data'}")
    print(M.groupby("kind")[["entanglement", "oracle_auc", "mf_agreement"]].describe().round(3).T.to_string())


def _job(args):
    out, name = args
    f = out / "runs" / f"{name}.csv"
    if f.exists():
        return name, "skip"
    meta = pd.read_csv(out / "TS_TEACHERS.csv").set_index("name").loc[name].to_dict()
    meta["name"] = name
    z = np.load(out / "data" / f"{name}.npz")
    t0 = time.time()
    R, Cr = run_dataset(z["X"], z["y"].astype(int), meta)
    R["seconds"] = time.time() - t0
    f.parent.mkdir(exist_ok=True)
    Cr.to_csv(out / "runs" / f"{name}_curve.csv", index=False)
    tmp = f.with_suffix(".tmp.csv"); R.to_csv(tmp, index=False); tmp.replace(f)
    return name, f"ok {time.time() - t0:.0f}s"


def do_run(out: Path, workers: int, only=None):
    M = pd.read_csv(out / "TS_TEACHERS.csv")
    am = out / "TS_AMENDMENT.json"
    if not am.exists():
        am.write_text(json.dumps({**AMENDMENT, "design": DESIGN,
                                  "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    names = list(M.name) if not only else [n for n in M.name if any(o in n for o in only)]
    # biggest p first
    names.sort(key=lambda n: -int(n.split("_p")[1].split("_")[0]))
    print(f"{len(names)} datasets", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, (out, n)): n for n in names}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, st = fu.result()
                print(f"[{i}/{len(names)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(names)}] FAILED {futs[fu]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry" if fails else ""))
    sys.exit(1 if fails else 0)


def _dstar(curve_means: pd.DataFrame, target: float, tol: float):
    """smallest D at which any family's mean AUC reaches target - tol (inf if none)"""
    ok = curve_means[curve_means.auc >= target - tol]
    return float(ok.D.min()) if len(ok) else np.inf


def do_score(out: Path, d=DESIGN):
    M = pd.read_csv(out / "TS_TEACHERS.csv")
    rows = []
    for nm in M.name:
        f = out / "runs" / f"{nm}.csv"
        if not f.exists():
            continue
        R = pd.read_csv(f)
        Cr = pd.read_csv(out / "runs" / f"{nm}_curve.csv")
        A = R.pivot_table(index="fold", columns="arm", values="auc")
        m = A.mean()
        cm = Cr.groupby(["family", "D"]).auc.mean().reset_index()
        tgt = max(m["S_J0"], m["S_MF"], m["S_Q"])      # best AQFM student
        own = {"J0": "S_J0", "MF": "S_MF", "Q": "S_Q"}[M.set_index("name").loc[nm, "kind"]]
        r = dict(name=nm, **{f"auc_{k}": v for k, v in m.items()},
                 gap_Q_MF=m["S_Q"] - m["S_MF"], gap_MF_J0=m["S_MF"] - m["S_J0"],
                 gap_Q_J0=m["S_Q"] - m["S_J0"], gap_Q_classical=m["S_Q"] - m["classical_best"],
                 pos_folds_Q_MF=int(((A["S_Q"] - A["S_MF"]) > 0).sum()),
                 own_student_best=bool(m[own] >= tgt - 0.01),
                 best_student=max(("S_J0", "S_MF", "S_Q"), key=lambda k: m[k]),
                 dstar=_dstar(cm, tgt, d["dstar_tol"]))
        for fam in cm.family.unique():
            r[f"dstar_{fam}"] = _dstar(cm[cm.family == fam], tgt, d["dstar_tol"])
        pk = cm.loc[cm.auc.idxmax()]          # post-hoc best point on the generic classical curves
        r.update(curve_peak_auc=pk.auc, curve_peak_D=pk.D, curve_peak_family=pk.family,
                 shortfall_generic=tgt - pk.auc, best_student_auc=tgt,
                 classical_choice=R[R.arm == "classical_best"].choice.str.replace(r"\d+$", "", regex=True).mode().iloc[0])
        if "auc_C1" in Cr:                    # the v1 setting, for comparison
            r["curve_peak_auc_C1"] = Cr.groupby(["family", "D"]).auc_C1.mean().max()
        rows.append(r)
    T = M.merge(pd.DataFrame(rows), on="name")
    T["dstar_over_p"] = T.dstar / T.p
    T.to_csv(out / "TS_DATASETS.csv", index=False)
    pd.set_option("display.width", 220)

    print(f"\n{len(T)} of {len(M)} datasets scored\n")
    g = (T.groupby(["kind", "kappa"])
          .agg(n=("name", "size"), E=("entanglement", "median"), oracle=("oracle_auc", "median"),
               S_J0=("auc_S_J0", "median"), S_MF=("auc_S_MF", "median"), S_Q=("auc_S_Q", "median"),
               classical=("auc_classical_best", "median"),
               gap_Q_MF=("gap_Q_MF", "median"), gap_MF_J0=("gap_MF_J0", "median"),
               dstar_over_p=("dstar_over_p", "median"),
               shortfall=("shortfall_generic", "median"),
               unreachable=("dstar", lambda v: float(np.isinf(v).mean()))))
    g.to_csv(out / "TS_SUMMARY.csv")
    print("=== by teacher kind x kappa (medians; AUC) ===")
    print(g.round(3).to_string())
    print("\n=== by teacher kind x readout ===")
    print(T.groupby(["kind", "readout"])[["gap_Q_MF", "gap_MF_J0", "dstar_over_p"]].median().round(3).to_string())

    print("\n=== hypothesis checks ===")
    print(f"H1 own student best (within 0.01): {T.own_student_best.mean():.0%} overall (pass >= 70%); by kind:",
          T.groupby("kind").own_student_best.mean().round(2).to_dict())
    print("   which student wins, by teacher kind:")
    print(pd.crosstab(T.kind, T.best_student).to_string())
    mf = T[T.kind == "MF"]
    if len(mf):
        a = (mf[mf.kappa >= 1].gap_MF_J0 >= 0.01).mean()
        print(f"H2 MF teachers, kappa>=1: S_MF - S_J0 >= 0.01 on {a:.0%} (pass >= 50%)")
    q = T[T.kind == "Q"]
    if len(q):
        b = (q[q.kappa <= 1].gap_Q_MF.abs() < 0.01).mean()
        print(f"H2b Q teachers, kappa<=1: |S_Q - S_MF| < 0.01 on {b:.0%} (pass >= 50%)")
    if len(q) > 3:
        rho = spearmanr(q.entanglement, q.gap_Q_MF).correlation
        print(f"H3 Spearman(E, Q-MF) over Q teachers: {rho:.2f}  (pass if > 0.3)")
        med = q.groupby("kappa").gap_Q_MF.median()
        print(f"H4 median Q-MF by kappa: {med.round(3).to_dict()}")
        print(f"R1 Q teachers with Q-MF < 0.01: {(q.gap_Q_MF < 0.01).mean():.0%}")
        rho6 = spearmanr(q.mf_agreement, q.gap_Q_MF).correlation
        print(f"H6 Spearman(mf_agreement, Q-MF): {rho6:.2f}  (pass if < -0.3)")
    print("H5 median D*/p and unreachable fraction by teacher kind:")
    print(T.groupby("kind").agg(dstar_over_p=("dstar_over_p", "median"),
                                unreachable=("dstar", lambda v: float(np.isinf(v).mean()))).round(3).to_string())

    print("\n=== classical cost detail ===")
    unr = lambda v: float(np.isinf(v).mean())
    print("D*/p (median) and unreachable fraction, by kind x kappa:")
    print(T.groupby(["kind", "kappa"]).agg(dstar_over_p=("dstar_over_p", "median"),
                                           unreachable=("dstar", unr),
                                           best_student=("best_student_auc", "median"),
                                           curve_peak=("curve_peak_auc", "median"),
                                           classical_best=("auc_classical_best", "median")).round(3).to_string())
    fams = [c for c in T.columns if c.startswith("dstar_") and c != "dstar_over_p"]
    print("\nunreachable fraction per classical family, by kind x readout:")
    print(T.groupby(["kind", "readout"])[fams].agg(unr).round(2).to_string())
    print("\nwhat classical_best picked (most common per dataset):")
    print(pd.crosstab(T.kind, T.classical_choice).to_string())
    if "curve_peak_auc_C1" in T:
        print(f"\ncurve peak, tuned C vs C=1 (median): {T.curve_peak_auc.median():.3f} vs {T.curve_peak_auc_C1.median():.3f}")
    print(f"\nwrote {out/'TS_DATASETS.csv'} and {out/'TS_SUMMARY.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["make", "run", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="substring filter on dataset names (run phase)")
    ap.add_argument("--ps", nargs="+", type=int, help="override DESIGN ps (make phase; for quick tests)")
    a = ap.parse_args()
    if a.ps:
        DESIGN["ps"] = a.ps
    if a.phase == "make":
        do_make(a.out)
    elif a.phase == "run":
        do_run(a.out, a.workers, a.only)
    else:
        do_score(a.out)


if __name__ == "__main__":
    main()
