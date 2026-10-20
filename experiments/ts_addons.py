#!/usr/bin/env python3
"""
Add-ons to an existing teacher-student run (tau sweep or scaling): no new
quantum simulation, only extra fitting on the saved labels and banks.

  nonlinear   strong nonlinear learners on the RAW inputs (standardized x):
              gradient boosting, RBF-kernel SVM, a small neural net.  Same outer
              folds as the main run (paired), hyperparameters by inner 3-fold CV.
              -> OUT/addons/nl/<name>.csv
  shots       the exact student S_Q re-fitted on SHOT-NOISY features:
              every expectation value <P> is replaced by its estimate from S
              single-shot measurements, 2*Binomial(S, (1+<P>)/2)/S - 1,
              for S in {100, 1000, 10000}.  Teachers (labels) and all classical
              opponents stay noiseless: only the hardware-measured arm gets noise.
              Same menu, folds and tuning as S_Q in the main run.
              -> OUT/addons/shots/<name>_S<S>.csv
  score       merges both with the main run's per-dataset table and reports the
              exact-map margin per region with the extended opponent set.

Usage (tau sweep):
  python experiments/ts_addons.py nonlinear --out $TT --workers 32
  python experiments/ts_addons.py shots     --out $TT --workers 32
  python experiments/ts_addons.py score     --out $TT
For the scaling run add:  --teachers SC_TEACHERS.csv --datasets SC_DATASETS.csv
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_student as ts            # noqa: E402
import ts_tau_sweep as tt               # noqa: E402
from aqfm_prescreen.aqfm import AQFM_FAMILIES, family_matrix   # noqa: E402

OUTER, INNER, CV_SEED = 5, 3, 42        # identical to the main runs
SHOTS = [100, 1000, 10000]


# ---------------------------------------------------------------------------
# nonlinear learners on raw inputs
# ---------------------------------------------------------------------------

def nl_grid(p):
    g = []
    for lr in (0.05, 0.1):
        for leaves in (8, 31):
            g.append(("HGB", f"lr{lr}_leaves{leaves}",
                      lambda lr=lr, leaves=leaves: HistGradientBoostingClassifier(
                          learning_rate=lr, max_leaf_nodes=leaves, max_iter=300, random_state=0)))
    for C in (0.3, 1.0, 3.0, 10.0, 30.0):
        for gm in (0.3, 1.0, 3.0):
            g.append(("SVM", f"C{C}_g{gm}/p",
                      lambda C=C, gm=gm: SVC(C=C, gamma=gm / p, kernel="rbf")))
    for alpha in (1e-4, 1e-2, 1.0):
        g.append(("MLP", f"64x64_a{alpha}",
                  lambda alpha=alpha: MLPClassifier((64, 64), alpha=alpha, max_iter=800,
                                                    early_stopping=True, random_state=0)))
    return g


def _score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def _fit_auc(make, Xa, ya, Xb, yb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make().fit(Xa, ya)
    return roc_auc_score(yb, _score(m, Xb))


def _folds(Z, y):
    return list(StratifiedKFold(OUTER, shuffle=True, random_state=CV_SEED).split(Z, y))


def nl_job(args):
    out, name, p = args
    f = out / "addons" / "nl" / f"{name}.csv"
    if f.exists():
        return name, "skip"
    _, Z = tt.inputs(p)
    y = np.load(out / "labels" / f"{name}.npy")
    grid = nl_grid(p)
    rows, t0 = [], time.time()
    for fo, (tr, te) in enumerate(_folds(Z, y)):
        Ztr, ytr = Z[tr], y[tr]
        inner = list(StratifiedKFold(INNER, shuffle=True, random_state=0).split(tr, ytr))
        inner_sc = {}
        for lrn, cfg, make in grid:
            inner_sc[(lrn, cfg)] = np.mean([_fit_auc(make, Ztr[a], ytr[a], Ztr[b], ytr[b]) for a, b in inner])
        best = {}
        for (lrn, cfg), s in inner_sc.items():
            if lrn not in best or s > best[lrn][1]:
                best[lrn] = (cfg, s)
        overall = max(best, key=lambda k: best[k][1])
        makers = {(l, c): mk for l, c, mk in grid}
        for lrn, (cfg, s) in best.items():
            auc = _fit_auc(makers[(lrn, cfg)], Ztr, ytr, Z[te], y[te])
            rows.append(dict(name=name, fold=fo, arm=lrn, choice=cfg, inner=s, auc=auc))
        cfg, s = best[overall]
        rows.append(dict(name=name, fold=fo, arm="nonlinear_best", choice=f"{overall}:{cfg}", inner=s,
                         auc=[r["auc"] for r in rows if r["fold"] == fo and r["arm"] == overall][0]))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(f)
    return name, f"ok {time.time() - t0:.0f}s"


# ---------------------------------------------------------------------------
# shot noise on the exact student's features
# ---------------------------------------------------------------------------

def shot_noisy(B, S, key):
    """estimate of each expectation from S single-shot +-1 outcomes"""
    rng = np.random.default_rng(int(hashlib.sha256(f"{key}_S{S}".encode()).hexdigest()[:8], 16))
    pr = np.clip((1.0 + np.asarray(B, float)) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(S, pr) / S - 1.0


_SH = {}


def _noisy_bank(out, p, kap, tau, S):
    k = (p, kap, tau, S)
    if k not in _SH:
        if len(_SH) > 400:
            _SH.clear()
        B = tt.load_bank(out, p, "Q", kap, tau)
        _SH[k] = shot_noisy(B, S, f"Q_p{p}_k{kap:g}_t{tau:g}")
    return _SH[k]


def shots_job(args):
    out, name, p, S, kappas, taus = args
    f = out / "addons" / "shots" / f"{name}_S{S}.csv"
    if f.exists():
        return name, "skip"
    _, Z = tt.inputs(p)
    y = np.load(out / "labels" / f"{name}.npy")
    menu = {f"{fam}|k{kap:g}|tau{tau:g}": (fam, kap, tau)
            for tau in taus for kap in kappas for fam in AQFM_FAMILIES}

    def F(key):
        fam, kap, tau = menu[key]
        return family_matrix(_noisy_bank(out, p, kap, tau, S), fam, p)

    rows, t0 = [], time.time()
    for fo, (tr, te) in enumerate(_folds(Z, y)):
        ytr = y[tr]
        inner = list(StratifiedKFold(INNER, shuffle=True, random_state=0).split(tr, ytr))
        paths = {}
        for key in menu:
            M = F(key)
            paths[key] = ts._cv_path([(M[tr][a], M[tr][b]) for a, b in inner], ytr, inner)
        k, c, s_in = ts._pick(paths)
        M = F(k)
        rows.append(dict(name=name, fold=fo, arm=f"S_Q_shots{S}", choice=k, C=c, inner=s_in,
                         auc=ts._outer(M[tr], M[te], ytr, y[te], c)))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(f)
    return name, f"ok {time.time() - t0:.0f}s"


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def _teachers(out, teachers):
    M = pd.read_csv(out / teachers)
    return M[~M.degenerate.astype(bool)]


def _pool(fn, jobs, workers, label):
    print(f"{len(jobs)} {label} jobs", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j[1] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, st = fu.result()
                print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(jobs)}] FAILED {futs[fu]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry" if fails else ""))
    return fails


def do_nonlinear(out, workers, teachers):
    M = _teachers(out, teachers)
    jobs = sorted([(out, n, int(p)) for n, p in zip(M.name, M.p)], key=lambda j: (-j[2], j[1]))
    return _pool(nl_job, jobs, workers, "nonlinear")


def _menu_from_design(out):
    """student kappa / tau grids of the run in `out` (read from its protocol)"""
    import json
    for nm in ("SC_PROTOCOL.json", "TT_PROTOCOL.json"):
        if (out / nm).exists():
            d = json.loads((out / nm).read_text())["design"]
            return d["student_kappas"], d["student_taus"]
    raise FileNotFoundError("no SC_PROTOCOL.json / TT_PROTOCOL.json in " + str(out))


def do_shots(out, workers, teachers, shots):
    M = _teachers(out, teachers)
    M = M[(M.kind == "Q") & (M.readout == "local")]
    kappas, taus = _menu_from_design(out)
    jobs = [(out, n, int(p), S, kappas, taus) for n, p in zip(M.name, M.p) for S in shots]
    jobs.sort(key=lambda j: (-j[2], j[1], j[3]))
    return _pool(shots_job, jobs, workers, "shot-noise")


def _region(t):
    if t.readout == "zlocal":
        return "Z-only"
    if t.kappa <= 0.5 or t.tau <= 1:
        return "below (weak coupling or short anneal)"
    if t.kappa >= 2 and 3 <= t.tau <= 20:
        return "window"
    if t.kappa == 1:
        return "transition (kappa=1)"
    return "other"


def do_score(out, teachers, datasets):
    T = pd.read_csv(out / datasets)
    nl, sh = [], []
    for nm in T.name:
        f = out / "addons" / "nl" / f"{nm}.csv"
        if f.exists():
            nl.append(pd.read_csv(f))
        for S in SHOTS:
            g = out / "addons" / "shots" / f"{nm}_S{S}.csv"
            if g.exists():
                sh.append(pd.read_csv(g))
    if nl:
        N = pd.concat(nl).pivot_table(index="name", columns="arm", values="auc", aggfunc="mean")
        T = T.merge(N.add_prefix("auc_").reset_index(), on="name", how="left")
    if sh:
        Sh = pd.concat(sh).pivot_table(index="name", columns="arm", values="auc", aggfunc="mean")
        T = T.merge(Sh.add_prefix("auc_").reset_index(), on="name", how="left")
    old_opp = ["auc_S_MF", "auc_S_J0", "auc_S_GS", "auc_classical_best"]
    new_opp = old_opp + (["auc_nonlinear_best"] if "auc_nonlinear_best" in T else [])
    T["margin_old"] = T.auc_S_Q - T[old_opp].max(axis=1)
    T["margin_new"] = T.auc_S_Q - T[new_opp].max(axis=1)
    for S in SHOTS:
        c = f"auc_S_Q_shots{S}"
        if c in T:
            T[f"margin_shots{S}"] = T[c] - T[new_opp].max(axis=1)
    T["region"] = T.apply(_region, axis=1)
    T.to_csv(out / "ADDONS_DATASETS.csv", index=False)

    pd.set_option("display.width", 250)
    q = T[T.kind == "Q"]
    print(f"{len(T)} datasets; nonlinear done for {int(T.get('auc_nonlinear_best', pd.Series(dtype=float)).notna().sum())}, "
          f"shots done for {int(T.get('auc_S_Q_shots1000', pd.Series(dtype=float)).notna().sum())}")
    if "auc_nonlinear_best" in T:
        print("\n=== exact teachers: margin of S_Q over the best opponent, without / with nonlinear learners ===")
        g = q.groupby("region").agg(n=("name", "size"),
                                    med_old=("margin_old", "median"), med_new=("margin_new", "median"),
                                    frac05_old=("margin_old", lambda v: (v >= 0.05).mean()),
                                    frac05_new=("margin_new", lambda v: (v.dropna() >= 0.05).mean()),
                                    max_new=("margin_new", "max"))
        print(g.round(3).to_string())
        print("\n=== median AUC of each opponent, exact teachers, by region ===")
        cols = [c for c in ["auc_S_Q", "auc_S_MF", "auc_classical_best", "auc_HGB", "auc_SVM", "auc_MLP",
                            "auc_nonlinear_best", "oracle_auc"] if c in q]
        print(q.groupby("region")[cols].median().round(3).to_string())
        w = q[q.region == "window"]
        if len(w):
            share = ((w[new_opp].max(axis=1) - 0.5) / (w.auc_S_Q - 0.5)).median()
            print(f"\nwindow: best opponent (incl. nonlinear) recovers {share:.0%} of the above-chance signal "
                  f"(median); nonlinear_best beats the old classical_best on "
                  f"{(w.auc_nonlinear_best > w.auc_classical_best).mean():.0%} of window datasets")
        mf = T[(T.kind == "MF") & (T.readout == "local")]
        if len(mf):
            print("\nMF teachers (classical generator), local: median AUC S_MF vs nonlinear_best vs classical_best by region")
            print(mf.groupby("region")[["auc_S_MF", "auc_nonlinear_best", "auc_classical_best"]].median().round(3).to_string())
    sc = [f"margin_shots{S}" for S in SHOTS if f"margin_shots{S}" in T]
    if sc:
        ql = q[q.readout == "local"]
        print("\n=== shot noise (exact student only): median margin over the best opponent, by region ===")
        print(ql.groupby("region")[["margin_new"] + sc].median().round(3).to_string())
        print("fraction with margin >= 0.05:")
        print(ql.groupby("region")[["margin_new"] + sc].agg(lambda v: (v.dropna() >= 0.05).mean() if v.notna().any() else np.nan).round(2).to_string())
        print("median S_Q AUC (noiseless vs shots):")
        print(ql.groupby("region")[["auc_S_Q"] + [f"auc_S_Q_shots{S}" for S in SHOTS if f'auc_S_Q_shots{S}' in T]]
              .median().round(3).to_string())
    print(f"\nwrote {out / 'ADDONS_DATASETS.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["nonlinear", "shots", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--teachers", default="TT_TEACHERS.csv")
    ap.add_argument("--datasets", default="TT_DATASETS.csv")
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--only", nargs="*", help="substring filter on dataset names (tests)")
    a = ap.parse_args()
    if a.only:
        global _teachers
        _orig = _teachers
        _teachers = lambda out, t: (lambda M: M[M.name.apply(lambda n: any(o in n for o in a.only))])(_orig(out, t))
    if a.phase == "nonlinear":
        sys.exit(1 if do_nonlinear(a.out, a.workers, a.teachers) else 0)
    elif a.phase == "shots":
        sys.exit(1 if do_shots(a.out, a.workers, a.teachers, a.shots) else 0)
    else:
        do_score(a.out, a.teachers, a.datasets)


if __name__ == "__main__":
    main()
