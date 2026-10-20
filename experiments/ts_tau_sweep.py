#!/usr/bin/env python3
"""
Teacher-student tau sweep: how does "who can learn it" change with the anneal
time tau?  Follow-up to teacher_student.py (v2), where tau=1 vs tau=5 decided
everything.

Expected regimes (anneal H(s) = (1-s)(-sum X) + s(-sum x_i Z_i - J sum Z_i Z_j),
s = t/tau, from |+>):
  small tau     sudden / perturbative: smooth label functions; classical features,
                mean-field and exact all agree
  intermediate  diabatic, crossing the Curie-Weiss transition (kappa >~ 1): labels
                oscillate in x; exact needed
  large tau     adiabatic: the state ends near the CLASSICAL ground state of the
                diagonal data Hamiltonian -> step-function labels that a classical
                argmin reproduces (the S_GS student); residue only from samples
                near level crossings

Differences from v2 (all recorded in TT_PROTOCOL.json):
  * 12 teacher taus (0.5 ... 40); the student tau grid is the same 12 values (well-specified)
  * ONE shared input matrix X per p (paired design): every dataset with the same
    p has the same inputs, so the student feature banks are computed once per p
    (`bank` work happens in `make`)
  * a 4th student S_GS: the ground-state configuration sigma(x) =
    argmin_sigma [-sum x_i sigma_i - J sum_{i<j} sigma_i sigma_j] (exact
    brute force over 2^p), features sigma_i and sigma_i sigma_j, for each kappa
  * readouts: local (random weights on <X_i>,<Y_i>,<Z_i>) and zlocal (random
    weights on <Z_i> only: what an annealer measures).  v2's `pairs` and
    `collect` are dropped (pairs was mostly a readout effect; collect = sum <X_i>
    is pure non-adiabatic residue at large tau)
  * everything else (soft labels, nested CV 5x3, tuned C, classical cost curves,
    leak-free classical_best) is v2's code

Phases:
  make   inputs, banks (parallel), labels, TT_TEACHERS.csv, TT_PROTOCOL.json
  run    one job per dataset -> OUT/runs/<name>.csv (+ _curve.csv); restartable
  score  TT_DATASETS.csv, TT_CURVES.csv (medians per kind/kappa/p/readout/tau),
         pre-registered checks
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
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_student as ts                                    # noqa: E402
from aqfm_prescreen.aqfm import (AQFM_FAMILIES, _bank_from_product_state, compute_bank,  # noqa: E402
                                 data_diagonal_batch, family_matrix, slices_for_p)
from aqfm_prescreen.dial import _global_z                       # noqa: E402

TAUS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0]

DESIGN = {
    **{k: ts.DESIGN[k] for k in ("n", "temp", "curve_mults", "poly_degrees", "spline_knots",
                                 "outer", "inner", "cv_seed", "dstar_tol", "C_grid")},
    "ps": [4, 6, 8],
    "kappas": [0.5, 1.0, 2.0, 4.0, 8.0],
    "teacher_taus": TAUS,
    "readouts": ["local", "zlocal"],
    "seeds_J0": 2,
    "seeds_coupled": 1,
    "student_kappas": [0.5, 1.0, 2.0, 4.0, 8.0],
    "student_taus": TAUS,
    "gs_families": ["Z", "ZZ", "Zfull"],
    "inputs": "one shared X per p, X ~ N(0,1), seed = sha256('tau_sweep_inputs_p{p}')",
    "version": "tau_sweep_1",
}
ts.DESIGN["C_grid"] = DESIGN["C_grid"]       # lr_path reads it from the v2 module

HYPOTHESES = {
    "T1_small_tau": "Q teachers with tau <= 1: |S_Q - S_MF| < 0.01 on >= 80%",
    "T2_nonmonotone": ("Q teachers with kappa >= 2 (pooled over p and readout): the median S_Q - S_MF "
                       "over tau peaks at an interior tau* < 40, and its value at tau = 40 is < 50% of the peak"),
    "T3_ground_state": ("Q teachers, zlocal readout, kappa >= 2, tau = 40: S_GS within 0.01 of the best "
                        "student on >= 50%"),
    "T4_sharpening": ("Q teachers with kappa >= 2: the width in log(tau) of the rise of median S_Q - S_MF "
                      "(from 25% to 75% of its peak) is smaller at p = 8 than at p = 4 "
                      "(exploratory: three sizes only)"),
    "T5_classical": ("median classical shortfall (best student - classical_best) is larger at tau = 5 than "
                     "at tau = 0.5 for both MF and Q teachers, and at every tau >= 3 the MF median "
                     "shortfall >= the Q median shortfall (replicates v2)"),
    "T6_predictor": "Q teachers: Spearman(mf_agreement, S_Q - S_MF) < -0.5",
    "R_report": "fraction of Q teachers with S_Q - S_MF < 0.01, per tau",
}

ARMS = ("S_J0", "S_MF", "S_Q", "S_GS")


# ---------------------------------------------------------------------------
# inputs, banks, ground state
# ---------------------------------------------------------------------------

def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def inputs(p, d=DESIGN):
    X = np.random.default_rng(_seed(f"tau_sweep_inputs_p{p}")).standard_normal((d["n"], p))
    return X, _global_z(X)


def bank_key(kind, kappa, tau):
    return f"{kind}_k{kappa:g}_t{tau:g}"


def spin_basis(p):
    b = np.arange(1 << p)
    return np.stack([1 - 2 * ((b >> q) & 1) for q in range(p)], axis=1).astype(float)


def ground_state_bank(Z, J):
    """Bank (same layout as compute_bank) of the classical ground state of
    H_data = -sum x_i Z_i - J sum_{i<j} Z_i Z_j: <Z_i> = sigma_i,
    <Z_i Z_j> = sigma_i sigma_j, all X / Y terms 0."""
    m, p = Z.shape
    ZS = spin_basis(p)
    sig = ZS[np.argmin(data_diagonal_batch(Z, J, ZS), axis=1)]
    loc = np.zeros((m, p, 3))
    loc[:, :, 2] = sig
    return _bank_from_product_state(loc)


def _bank_job(args):
    out, p, kind, kappa, tau = args
    f = out / "banks" / f"p{p}" / f"{bank_key(kind, kappa, tau)}.npy"
    if f.exists():
        return f.name, "skip"
    _, Z = inputs(p)
    J = 0.0 if kind == "J0" else kappa / max(p - 1, 1)
    t0 = time.time()
    if kind == "GS":
        B = ground_state_bank(Z, J)
    else:
        B = compute_bank(Z, J, tau, method="meanfield" if kind == "MF" else "strang")
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, np.asarray(B, dtype=np.float64))
    tmp.replace(f)
    return f.name, f"ok {time.time() - t0:.0f}s"


def bank_specs(d=DESIGN):
    """every bank any teacher or student needs"""
    taus = sorted(set(d["teacher_taus"]) | set(d["student_taus"]))
    kaps = sorted(set(d["kappas"]) | set(d["student_kappas"]))
    S = []
    for p in d["ps"]:
        S += [(p, "J0", 0.0, t) for t in taus]
        S += [(p, k, kap, t) for k in ("MF", "Q") for kap in kaps for t in taus]
        S += [(p, "GS", kap, 0.0) for kap in kaps]
    return S


def load_bank(out, p, kind, kappa, tau):
    return np.load(out / "banks" / f"p{p}" / f"{bank_key(kind, kappa, tau)}.npy")


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------

def readout_matrix(B, readout, p):
    if readout == "local":
        return family_matrix(B, "XYZ", p)
    if readout == "zlocal":
        return family_matrix(B, "Z", p)
    raise ValueError(readout)


def teacher_specs(d=DESIGN):
    specs = []
    for p, tau, ro in product(d["ps"], d["teacher_taus"], d["readouts"]):
        for s in range(d["seeds_J0"]):
            specs.append(dict(kind="J0", p=p, kappa=0.0, tau=tau, readout=ro, seed=s))
        for kind, kap in product(("MF", "Q"), d["kappas"]):
            for s in range(d["seeds_coupled"]):
                specs.append(dict(kind=kind, p=p, kappa=kap, tau=tau, readout=ro, seed=s))
    for sp in specs:
        sp["name"] = f"tt_{sp['kind']}_p{sp['p']}_k{sp['kappa']:g}_t{sp['tau']:g}_{sp['readout']}_s{sp['seed']}"
        sp["data_seed"] = _seed(sp["name"])
    return specs


def _score_from(F, w):
    sd = F.std(0)
    keep = sd > 1e-9                       # constant columns carry no signal
    if not keep.any():
        return None
    return ts._std(((F[:, keep] - F[:, keep].mean(0)) / sd[keep]) @ w[keep])


def make_labels(out, sp, d=DESIGN):
    rng = np.random.default_rng(sp["data_seed"])
    p = sp["p"]
    B = load_bank(out, p, sp["kind"], sp["kappa"], sp["tau"])
    F = readout_matrix(B, sp["readout"], p)
    w = rng.standard_normal(F.shape[1])
    score = _score_from(F, w)
    if score is None:
        return None, dict(sp, degenerate=True)
    prob = 1.0 / (1.0 + np.exp(-score / d["temp"]))
    y = (rng.random(d["n"]) < prob).astype(int)
    meta = dict(sp, J=0.0 if sp["kind"] == "J0" else sp["kappa"] / max(p - 1, 1),
                entanglement=ts.entanglement(B, p), oracle_auc=roc_auc_score(y, score),
                pos_rate=float(y.mean()), degenerate=False)
    for twin, key in (("MF", "mf_agreement"), ("GS", "gs_agreement")):
        if sp["kind"] == "J0":
            meta[key] = np.nan
            continue
        Bt = load_bank(out, p, twin, sp["kappa"], sp["tau"] if twin == "MF" else 0.0)
        st = _score_from(readout_matrix(Bt, sp["readout"], p), w)
        meta[key] = float(np.corrcoef(score, st)[0, 1]) if st is not None else np.nan
    return y, meta


# ---------------------------------------------------------------------------
# students (lazy: matrices are built on demand from the shared banks)
# ---------------------------------------------------------------------------

class LazySets(Mapping):
    def __init__(self, makers):
        self._m = makers

    def __getitem__(self, k):
        return self._m[k]()

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)


_CACHE = {}


def _student_sets(out, p, d=DESIGN):
    if _CACHE.get("p") == p:
        return _CACHE["S"], _CACHE["C"]
    _CACHE.clear()
    banks = {}

    def bank(kind, kap, tau):
        k = (kind, kap, tau)
        if k not in banks:
            banks[k] = load_bank(out, p, kind, kap, tau)
        return banks[k]

    S = {a: {} for a in ARMS}
    for tau in d["student_taus"]:
        for fam in AQFM_FAMILIES:
            S["S_J0"][f"{fam}|tau{tau:g}"] = (lambda fam=fam, tau=tau: family_matrix(bank("J0", 0.0, tau), fam, p))
        for kap in d["student_kappas"]:
            for arm, kind in (("S_MF", "MF"), ("S_Q", "Q")):
                for fam in AQFM_FAMILIES:
                    S[arm][f"{fam}|k{kap:g}|tau{tau:g}"] = (
                        lambda fam=fam, kap=kap, tau=tau, kind=kind: family_matrix(bank(kind, kap, tau), fam, p))
    sl, _ = slices_for_p(p)
    for kap in d["student_kappas"]:
        for fam in d["gs_families"]:
            cols = {"Z": [sl["Z"]], "ZZ": [sl["ZZ"]], "Zfull": [sl["Z"], sl["ZZ"]]}[fam]
            S["S_GS"][f"{fam}|k{kap:g}"] = (
                lambda cols=cols, kap=kap: np.hstack([bank("GS", kap, 0.0)[:, c] for c in cols]))
    S = {a: LazySets(m) for a, m in S.items()}
    X, Z = inputs(p)
    C = ts.curve_sets(Z, d)
    _CACHE.update(p=p, S=S, C=C, X=X)
    return S, C


def _job(args):
    out, name, p = args
    f = out / "runs" / f"{name}.csv"
    if f.exists():
        return name, "skip"
    S, C = _student_sets(out, p)
    y = np.load(out / "labels" / f"{name}.npy")
    t0 = time.time()
    R, Cr = ts.run_dataset(_CACHE["X"], y, {"name": name}, DESIGN, S=S, C=C)
    R["seconds"] = time.time() - t0
    f.parent.mkdir(exist_ok=True)
    Cr.to_csv(out / "runs" / f"{name}_curve.csv", index=False)
    tmp = f.with_suffix(".tmp.csv")
    R.to_csv(tmp, index=False)
    tmp.replace(f)
    return name, f"ok {time.time() - t0:.0f}s"


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def do_make(out: Path, workers: int, d=DESIGN):
    out.mkdir(parents=True, exist_ok=True)
    proto = {"design": d, "hypotheses": HYPOTHESES, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "note": "written by `make`, before any student was run; follows teacher_student v2 "
                     "(TS_AMENDMENT.json there)"}
    (out / "TT_PROTOCOL.json").write_text(json.dumps(proto, indent=2))
    for p in d["ps"]:
        (out / "banks" / f"p{p}").mkdir(parents=True, exist_ok=True)
    specs = bank_specs(d)
    specs.sort(key=lambda s: -(s[0] * 100 + s[3] * (s[1] == "Q")))      # slow exact banks first
    print(f"{len(specs)} banks", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_bank_job, (out,) + s): s for s in specs}
        for i, fu in enumerate(as_completed(futs), 1):
            nm, st = fu.result()
            if i % 50 == 0 or i == len(specs):
                print(f"[bank {i}/{len(specs)} {time.time() - t0:5.0f}s] {st} {nm}", flush=True)
    (out / "labels").mkdir(exist_ok=True)
    metas = []
    for sp in teacher_specs(d):
        y, meta = make_labels(out, sp, d)
        if y is not None:
            np.save(out / "labels" / f"{sp['name']}.npy", y)
        metas.append(meta)
    M = pd.DataFrame(metas)
    M.to_csv(out / "TT_TEACHERS.csv", index=False)
    ok = M[~M.degenerate]
    print(f"made {len(ok)} datasets ({int(M.degenerate.sum())} degenerate, skipped)")
    pd.set_option("display.width", 200)
    print(ok.groupby(["kind", "tau"])[["entanglement", "oracle_auc", "mf_agreement", "gs_agreement"]]
            .median().round(3).unstack(0).to_string())


def do_run(out: Path, workers: int, only=None):
    M = pd.read_csv(out / "TT_TEACHERS.csv")
    M = M[~M.degenerate]
    jobs = [(out, n, int(p)) for n, p in zip(M.name, M.p) if not only or any(o in n for o in only)]
    jobs.sort(key=lambda j: (-j[2], j[1]))            # big p first; same p together (bank cache)
    print(f"{len(jobs)} datasets", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, j): j[1] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, st = fu.result()
                print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(jobs)}] FAILED {futs[fu]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry" if fails else ""))
    sys.exit(1 if fails else 0)


def _rise_width(tau, gap):
    """width in log(tau) of the rise from 25% to 75% of the peak (linear interp in log tau)"""
    lt, g = np.log(np.asarray(tau, float)), np.asarray(gap, float)
    if len(g) < 3:
        return np.nan
    pk = int(np.argmax(g))
    if g[pk] <= 0.01:
        return np.nan
    def cross(level):
        for i in range(1, pk + 1):
            if g[i] >= level > g[i - 1]:
                return lt[i - 1] + (level - g[i - 1]) / (g[i] - g[i - 1]) * (lt[i] - lt[i - 1])
        return lt[0] if g[0] >= level else np.nan
    return float(cross(0.75 * g[pk]) - cross(0.25 * g[pk]))


def do_score(out: Path, d=DESIGN):
    M = pd.read_csv(out / "TT_TEACHERS.csv")
    M = M[~M.degenerate]
    rows = []
    for nm, kind in zip(M.name, M.kind):
        f = out / "runs" / f"{nm}.csv"
        if not f.exists():
            continue
        R = pd.read_csv(f)
        Cr = pd.read_csv(out / "runs" / f"{nm}_curve.csv")
        A = R.pivot_table(index="fold", columns="arm", values="auc")
        m = A.mean()
        cm = Cr.groupby(["family", "D"]).auc.mean().reset_index()
        tgt = max(m[a] for a in ARMS)
        own = {"J0": "S_J0", "MF": "S_MF", "Q": "S_Q"}[kind]
        pk = cm.loc[cm.auc.idxmax()]
        r = dict(name=nm, **{f"auc_{k}": v for k, v in m.items()},
                 gap_Q_MF=m["S_Q"] - m["S_MF"], gap_MF_J0=m["S_MF"] - m["S_J0"],
                 gap_Q_GS=m["S_Q"] - m["S_GS"], best_student_auc=tgt,
                 best_student=max(ARMS, key=lambda k: m[k]),
                 own_student_best=bool(m[own] >= tgt - 0.01),
                 gs_within=bool(m["S_GS"] >= tgt - 0.01),
                 shortfall_classical=tgt - m["classical_best"],
                 curve_peak_auc=pk.auc, curve_peak_family=pk.family, curve_peak_D=pk.D,
                 dstar=ts._dstar(cm, tgt, d["dstar_tol"]),
                 classical_choice=R[R.arm == "classical_best"].choice.str.replace(r"\d+$", "", regex=True).mode().iloc[0])
        rows.append(r)
    T = M.merge(pd.DataFrame(rows), on="name")
    T["dstar_over_p"] = T.dstar / T.p
    T.to_csv(out / "TT_DATASETS.csv", index=False)
    vals = ["entanglement", "oracle_auc", "mf_agreement", "gs_agreement"] + [f"auc_{a}" for a in ARMS] + \
           ["auc_classical_best", "gap_Q_MF", "gap_MF_J0", "gap_Q_GS", "shortfall_classical", "dstar_over_p"]
    curves = T.groupby(["kind", "kappa", "p", "readout", "tau"])[vals].median().reset_index()
    curves.to_csv(out / "TT_CURVES.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\n{len(T)} of {len(M)} datasets scored\n")
    q, mf = T[T.kind == "Q"], T[T.kind == "MF"]
    print("=== median S_Q - S_MF  (Q teachers; rows kappa, columns tau) ===")
    print(q.pivot_table(index="kappa", columns="tau", values="gap_Q_MF", aggfunc="median").round(3).to_string())
    print("\n=== median classical shortfall (best student - classical_best), rows kind, columns tau ===")
    print(T.pivot_table(index="kind", columns="tau", values="shortfall_classical", aggfunc="median").round(3).to_string())
    print("\n=== median AUC by student, Q teachers kappa >= 2, columns tau ===")
    print(q[q.kappa >= 2].groupby("tau")[[f"auc_{a}" for a in ARMS] + ["auc_classical_best", "oracle_auc", "entanglement"]]
           .median().T.round(3).to_string())
    print("\n=== same, MF teachers kappa >= 2 ===")
    print(mf[mf.kappa >= 2].groupby("tau")[[f"auc_{a}" for a in ARMS] + ["auc_classical_best", "oracle_auc"]]
            .median().T.round(3).to_string())
    print("\n=== which student wins (count), Q teachers, rows tau ===")
    print(pd.crosstab(q.tau, q.best_student).to_string())

    print("\n=== pre-registered checks ===")
    def _checks():
        a = (q[q.tau <= 1].gap_Q_MF.abs() < 0.01).mean()
        print(f"T1 Q, tau<=1: |S_Q-S_MF|<0.01 on {a:.0%}  (pass >= 80%)")
        g = q[q.kappa >= 2].groupby("tau").gap_Q_MF.median()
        tstar, peak, last = g.idxmax(), g.max(), g.loc[max(g.index)]
        print(f"T2 Q kappa>=2 median gap by tau: {g.round(3).to_dict()}\n   peak {peak:.3f} at tau*={tstar:g}; "
              f"tau={max(g.index):g}: {last:.3f} ({last / peak:.0%} of peak)  -> {'PASS' if tstar < max(g.index) and last < 0.5 * peak else 'FAIL'}")
        q20 = q[(q.readout == "zlocal") & (q.kappa >= 2) & (q.tau == max(d['teacher_taus']))]
        if len(q20):
            print(f"T3 Q zlocal kappa>=2 tau={max(d['teacher_taus']):g}: S_GS within 0.01 of best on {q20.gs_within.mean():.0%} (pass >= 50%)")
        print("T4 rise width in log(tau), Q kappa>=2, by p:")
        for p in d["ps"]:
            gp = q[(q.kappa >= 2) & (q.p == p)].groupby("tau").gap_Q_MF.median()
            print(f"   p={p}: width {_rise_width(gp.index, gp.values):.2f}   gaps {gp.round(3).to_dict()}")
        sh = T.pivot_table(index="kind", columns="tau", values="shortfall_classical", aggfunc="median")
        t5a = all(sh.loc[k, 5.0] > sh.loc[k, 0.5] for k in ("MF", "Q"))
        t5b = all(sh.loc["MF", t] >= sh.loc["Q", t] for t in sh.columns if t >= 3)
        print(f"T5 classical shortfall rises 0.5->5 for MF and Q: {t5a}; MF >= Q at every tau>=3: {t5b}")
        rho = spearmanr(q.mf_agreement, q.gap_Q_MF, nan_policy="omit").correlation
        print(f"T6 Spearman(mf_agreement, Q-MF) over Q teachers: {rho:.2f}  (pass < -0.5)")
        print("R  Q teachers with Q-MF < 0.01, by tau:", (q.gap_Q_MF < 0.01).groupby(q.tau).mean().round(2).to_dict())

    try:
        _checks()
    except Exception as e:     # partial runs (tests): report what is missing, keep the tables
        print(f"   (checks incomplete: {type(e).__name__}: {e})")
    print(f"\nwrote {out / 'TT_DATASETS.csv'} and {out / 'TT_CURVES.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["make", "run", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="substring filter on dataset names (run phase)")
    ap.add_argument("--ps", nargs="+", type=int, help="override ps (quick tests)")
    ap.add_argument("--taus", nargs="+", type=float, help="override teacher AND student taus (quick tests)")
    a = ap.parse_args()
    if a.ps:
        DESIGN["ps"] = a.ps
    if a.taus:
        DESIGN["teacher_taus"] = DESIGN["student_taus"] = a.taus
    if a.phase == "make":
        do_make(a.out, a.workers)
    elif a.phase == "run":
        do_run(a.out, a.workers, a.only)
    else:
        do_score(a.out)


if __name__ == "__main__":
    main()
