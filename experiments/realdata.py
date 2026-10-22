#!/usr/bin/env python3
"""
The ladder and the feature count on real tabular datasets (docs/realdata_plan.md).

Phases, per dataset (data/ophtho/<name>/, adapters in real_datasets.py):

  prepare  clean -> split 50/50 into DISCOVERY and CONFIRMATION halves (stratified,
           grouped by patient where there are groups) -> impute with discovery
           medians -> choose the p <= 8 qubit features (random-forest importance on
           the discovery half) -> standardize with discovery statistics, clip +-3.
           -> OUT/<name>/prepared.npz, PREP.json
  banks    simulate every feature bank the students need (exact, mean-field, J=0,
           ground state) on ALL rows of the prepared inputs (row-by-row, nothing
           fitted).  -> OUT/<name>/banks/*.npy
  run      DISCOVERY half only: all tau-sweep students (nested CV 5x3, grouped
           where needed), nonlinear learners on the p inputs, the reference
           nonlinear learners on ALL features, and the screen measures.
           -> OUT/<name>/discovery_students.csv, discovery_curve.csv,
              discovery_nonlinear.csv, discovery_measures.csv
  confirm  every student tuned by 5-fold CV on the whole discovery half, refit on
           it, scored ONCE on the confirmation half.  -> OUT/<name>/confirmation.npz
  score    verdicts, feature counts, confirmation claims with bootstrap intervals,
           checks R1-R5.  -> OUT/REAL_DATASETS.csv, OUT/REAL_RESULTS.csv

Usage (from the repo root):
  python experiments/realdata.py status
  python experiments/realdata.py prepare --out runs/realdata
  python experiments/realdata.py banks   --out runs/realdata --workers 32
  python experiments/realdata.py run     --out runs/realdata --workers 8
  python experiments/realdata.py confirm --out runs/realdata --workers 8
  python experiments/realdata.py score   --out runs/realdata
Add --group pool to run the 55 Run-2 datasets instead (output runs/realdata_pool), and
--only NAME [NAME ...] to restrict to some datasets.  Every phase skips work
that is already done; nothing outside OUT and data/ophtho is read or written.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import real_datasets as RD              # noqa: E402
import screen_measures as SM            # noqa: E402
import teacher_student as ts            # noqa: E402
import ts_addons as ad                  # noqa: E402
import ts_tau_sweep as tt               # noqa: E402
from aqfm_prescreen.aqfm import AQFM_FAMILIES, compute_bank, family_matrix, slices_for_p   # noqa: E402

D_STUDENTS = tt.DESIGN                  # the students are exactly the tau sweep's
DESIGN = {
    "version": "realdata_1",
    "plan": "docs/realdata_plan.md",
    "datasets": ["dr_debrecen", "papila", "myopia", "grape", "keratoconus", "perg_ioba", "fives_vascx"],
    "pool": RD.POOL,                    # Amendment 3: the 55 Run-2 datasets (--group pool)
    "p_max": 8, "p_min": 2,             # Amendment 3: 2 (was 4; no effect on the 5 ophthalmic runs, all p = 8)
    "missing_max": 0.20,
    "rf_trees": 500, "rf_seed": 0,
    "clip": 3.0,
    "n_boot": 1000,
    "verdict_tol": 0.01,                # "within 0.01 of the exact map"
    "win_margin": 0.05,
    "compact_gap": 0.02,
}
CLASSICAL = ["auc_classical_best", "auc_nonlinear_best"]
STANDINS = ["auc_S_MF", "auc_S_J0", "auc_S_GS"]


def _seed(s):
    return tt._seed(s)


# ---------------------------------------------------------------------------
# folds (one helper, so every phase uses identical folds)
# ---------------------------------------------------------------------------

def outer_folds(y, groups=None, k=None, seed=None):
    """the students' outer folds on the rows given (grouped when groups is not None)"""
    k = k or D_STUDENTS["outer"]
    seed = D_STUDENTS["cv_seed"] if seed is None else seed
    Z0 = np.zeros((len(y), 1))
    if groups is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(Z0, y))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(Z0, y, groups))


def inner_folds(ytr, gtr=None):
    Z0 = np.zeros((len(ytr), 1))
    if gtr is None:
        return list(StratifiedKFold(D_STUDENTS["inner"], shuffle=True, random_state=0).split(Z0, ytr))
    return list(StratifiedGroupKFold(D_STUDENTS["inner"], shuffle=True, random_state=0).split(Z0, ytr, gtr))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def clean_features(X, missing_max):
    """drop mostly-missing columns; 0/1-encode binary categoricals; one-hot the rest"""
    X = X.copy()
    dropped = {c: "more than 20% missing" for c in X if X[c].isna().mean() > missing_max}
    X = X.drop(columns=list(dropped))
    out = []
    for c in X:
        v = X[c]
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().sum() == v.notna().sum():                     # numeric column
            out.append(num.rename(c))
            continue
        levels = sorted(v.dropna().astype(str).unique())
        if len(levels) <= 2:
            out.append(v.map(lambda x, L=levels: np.nan if pd.isna(x) else float(str(x) == L[-1])).rename(f"{c}={levels[-1]}"))
        else:
            for L in levels:
                out.append(v.map(lambda x, L=L: np.nan if pd.isna(x) else float(str(x) == L)).rename(f"{c}={L}"))
    return pd.concat(out, axis=1) if out else pd.DataFrame(index=X.index), dropped


def split_halves(y, groups, name):
    """50/50 split, stratified by label, grouped by patient where there are groups.
    Returns a boolean mask: True = discovery."""
    Z0 = np.zeros((len(y), 1))
    if groups is None:
        tr, te = next(StratifiedKFold(2, shuffle=True, random_state=_seed(f"split_{name}")).split(Z0, y))
    else:
        tr, te = next(StratifiedGroupKFold(2, shuffle=True, random_state=_seed(f"split_{name}")).split(Z0, y, groups))
    disc = np.zeros(len(y), bool)
    disc[tr] = True
    return disc


def choose_qubits(Xd, yd, d=DESIGN):
    """the p features that become qubits, chosen on the DISCOVERY half only"""
    cols = list(Xd.columns)
    rf = RandomForestClassifier(d["rf_trees"], random_state=d["rf_seed"], n_jobs=1).fit(Xd.to_numpy(), yd)
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    uni = pd.Series({c: abs(roc_auc_score(yd, Xd[c]) - 0.5) for c in cols}).sort_values(ascending=False)
    chosen = cols if len(cols) <= d["p_max"] else list(imp.index[: d["p_max"]])
    return chosen, imp, uni


def prepare(name, out, d=DESIGN):
    """-> dict with everything later phases need (also saved to disk)"""
    raw = RD.load(name)
    X, dropped = clean_features(raw.X, d["missing_max"])
    y = np.asarray(raw.y).astype(int)
    groups = None if raw.groups is None else np.asarray(raw.groups)
    disc = split_halves(y, groups, name)
    med = X[disc].median()
    X = X.fillna(med)
    const = [c for c in X if X.loc[disc, c].std() == 0]          # carries nothing on the discovery half
    X = X.drop(columns=const)
    dropped.update({c: "constant on the discovery half" for c in const})
    info = dict(name=name, n=len(y), positives=int(y.sum()), n_disc=int(disc.sum()), n_conf=int((~disc).sum()),
                pos_disc=int(y[disc].sum()), pos_conf=int(y[~disc].sum()), grouped=groups is not None,
                n_features=X.shape[1], dropped=dropped, adapter=raw.info, design=d)
    if X.shape[1] < d["p_min"]:
        info.update(status="dropped", reason=f"only {X.shape[1]} usable features (< {d['p_min']})")
        return info
    if min(info["pos_disc"], info["pos_conf"], info["n_disc"] - info["pos_disc"], info["n_conf"] - info["pos_conf"]) < 10:
        info.update(status="dropped", reason="fewer than 10 of a class in one half")
        return info
    chosen, imp, uni = choose_qubits(X[disc], y[disc], d)
    mu, sd = X.loc[disc, chosen].mean(), X.loc[disc, chosen].std(ddof=0)
    Z = np.clip(((X[chosen] - mu) / sd).to_numpy(float), -d["clip"], d["clip"])
    mu_a, sd_a = X[disc].mean(), X[disc].std(ddof=0)
    Xall = ((X - mu_a) / sd_a).to_numpy(float)          # reference learners: all features, same scaling rule (unclipped)
    info.update(status="ok", p=len(chosen), qubits=chosen,
                rf_importance=imp.round(5).to_dict(), univariate_auc_gap=uni.round(5).to_dict())
    o = out / name
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "prepared.npz", Z=Z, Xall=Xall, y=y, disc=disc,
                        groups=np.arange(len(y)) if groups is None else pd.factorize(groups)[0],
                        grouped=groups is not None)
    return info


def do_prepare(out, only):
    out.mkdir(parents=True, exist_ok=True)
    for name in only:
        o = out / name / "PREP.json"
        if o.exists() and json.loads(o.read_text())["status"] in ("ok", "dropped"):
            print(f"[skip] {name}: already prepared")
            continue
        try:
            info = prepare(name, out)
        except RD.NotReady as e:
            # recorded so the results table lists it with the reason (plan Section 9);
            # retried on the next `prepare` in case the files have arrived
            print(f"[not ready] {name}: {e}")
            (out / name).mkdir(parents=True, exist_ok=True)
            o.write_text(json.dumps(dict(name=name, status="not obtained", reason=str(e).splitlines()[0],
                                         n=None, positives=None), indent=2))
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(info, indent=2, default=str))
        if info["status"] == "ok":
            print(f"[ok] {name}: n={info['n']} ({info['positives']} pos), discovery {info['n_disc']} / "
                  f"confirmation {info['n_conf']}, p={info['p']} qubits = {info['qubits']}")
        else:
            print(f"[dropped] {name}: {info['reason']}")


def load_prepared(out, name):
    P = np.load(out / name / "prepared.npz")
    info = json.loads((out / name / "PREP.json").read_text())
    groups = P["groups"] if bool(P["grouped"]) else None
    return P["Z"], P["Xall"], P["y"], P["disc"], groups, info


def ready(out, only, phase_file=None):
    names = []
    for name in only:
        f = out / name / "PREP.json"
        if f.exists() and json.loads(f.read_text())["status"] == "ok":
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------

def bank_specs(d=D_STUDENTS):
    S = [("J0", 0.0, t) for t in d["student_taus"]]
    S += [(k, kap, t) for k in ("MF", "Q") for kap in d["student_kappas"] for t in d["student_taus"]]
    S += [("GS", kap, 0.0) for kap in d["student_kappas"]]
    return S


def _bank_job(args):
    out, name, kind, kappa, tau = args
    f = out / name / "banks" / f"{tt.bank_key(kind, kappa, tau)}.npy"
    if f.exists():
        return name, "skip"
    Z = np.load(out / name / "prepared.npz")["Z"]
    p = Z.shape[1]
    J = 0.0 if kind == "J0" else kappa / max(p - 1, 1)
    t0 = time.time()
    B = tt.ground_state_bank(Z, J) if kind == "GS" else \
        compute_bank(Z, J, tau, method="meanfield" if kind == "MF" else "strang")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, np.asarray(B, dtype=np.float64))
    tmp.replace(f)
    return f"{name}/{f.name}", f"ok {time.time() - t0:.0f}s"


def do_banks(out, workers, only):
    jobs = [(out, n, *s) for n in ready(out, only) for s in bank_specs()]
    jobs.sort(key=lambda j: -(j[4] * (j[2] == "Q")))            # slow exact banks first
    return ad._pool(_bank_job, jobs, workers, "bank")


def banks_complete(out, name):
    """every expected bank file exists (half-written .tmp files don't count)"""
    b = out / name / "banks"
    return all((b / f"{tt.bank_key(*s)}.npy").exists() for s in bank_specs())


def student_sets(out, name, p, rows, d=D_STUDENTS):
    """the tau sweep's four map students, restricted to `rows` (lazy, built on demand)"""
    cache = {}

    def bank(kind, kap, tau):
        k = (kind, kap, tau)
        if k not in cache:
            cache[k] = np.load(out / name / "banks" / f"{tt.bank_key(kind, kap, tau)}.npy")[rows]
        return cache[k]

    S = {a: {} for a in tt.ARMS}
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
            S["S_GS"][f"{fam}|k{kap:g}"] = (lambda cols=cols, kap=kap: np.hstack([bank("GS", kap, 0.0)[:, c] for c in cols]))
    return {a: tt.LazySets(m) for a, m in S.items()}


def family_size(choice, p):
    """number of measured features in a map student's choice key, e.g. 'XYZ|k2|tau5' -> 3p"""
    fam = choice.split("|")[0]
    q = p * (p - 1) // 2
    return {"X": p, "Y": p, "Z": p, "XYZ": 3 * p, "XX": q, "YY": q, "ZZ": q, "pairs": 3 * q,
            "full": 3 * p + 3 * q, "Zfull": p + q}[fam]


# ---------------------------------------------------------------------------
# run (discovery half)
# ---------------------------------------------------------------------------

def nonlinear_cv(Z, y, groups, tag, name):
    """the tau-sweep nonlinear learners (same grids), nested CV with our folds"""
    grid = ad.nl_grid(Z.shape[1])
    makers = {(l, c): mk for l, c, mk in grid}
    rows = []
    for fo, (tr, te) in enumerate(outer_folds(y, groups)):
        Ztr, ytr = Z[tr], y[tr]
        inner = inner_folds(ytr, None if groups is None else groups[tr])
        sc = {(l, c): np.mean([ad._fit_auc(mk, Ztr[a], ytr[a], Ztr[b], ytr[b]) for a, b in inner]) for l, c, mk in grid}
        best = {}
        for (l, c), s in sc.items():
            if l not in best or s > best[l][1]:
                best[l] = (c, s)
        for l, (c, s) in best.items():
            rows.append(dict(name=name, fold=fo, arm=f"{tag}{l}", choice=c, inner=s,
                             auc=ad._fit_auc(makers[(l, c)], Ztr, ytr, Z[te], y[te])))
        top = max(best, key=lambda k: best[k][1])
        rows.append(dict(name=name, fold=fo, arm=f"{tag}nonlinear_best", choice=f"{top}:{best[top][0]}",
                         inner=best[top][1], auc=[r["auc"] for r in rows if r["fold"] == fo and r["arm"] == f"{tag}{top}"][0]))
    return rows


def _run_job(args):
    out, name = args
    o = out / name
    if (o / "discovery_students.csv").exists():
        return name, "skip"
    Z, Xall, y, disc, groups, info = load_prepared(out, name)
    p = Z.shape[1]
    rows = np.flatnonzero(disc)
    gd = None if groups is None else groups[rows]
    t0 = time.time()
    S = student_sets(out, name, p, rows)
    C_all = ts.curve_sets(Z, D_STUDENTS)                       # label-free transforms, all rows
    C = {k: F[rows] for k, F in C_all.items()}
    R, Cr = ts.run_dataset(Z[rows], y[rows], {"name": name}, D_STUDENTS, S=S, C=C, groups=gd)
    N = pd.DataFrame(nonlinear_cv(Z[rows], y[rows], gd, "", name)
                     + nonlinear_cv(Xall[rows], y[rows], gd, "ref_all_", name))
    M = pd.DataFrame([SM.all_measures(Z[rows], y[rows], name, folds=outer_folds(y[rows], gd))])
    Cr.to_csv(o / "discovery_curve.csv", index=False)
    N.to_csv(o / "discovery_nonlinear.csv", index=False)
    M.to_csv(o / "discovery_measures.csv", index=False)
    tmp = o / "discovery_students.tmp.csv"
    R.to_csv(tmp, index=False)
    tmp.replace(o / "discovery_students.csv")
    return name, f"ok {time.time() - t0:.0f}s"


def do_run(out, workers, only):
    names = [n for n in ready(out, only) if banks_complete(out, n)]
    missing = set(ready(out, only)) - set(names)
    if missing:
        print(f"banks incomplete for {sorted(missing)}: run the banks phase first")
    return ad._pool(_run_job, [(out, n) for n in names], workers, "discovery")


# ---------------------------------------------------------------------------
# confirm (one scoring on the confirmation half)
# ---------------------------------------------------------------------------

def _fit_scores(Ftr, ytr, Fte, C):
    sc = StandardScaler().fit(Ftr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Ftr), ytr)
    return m.decision_function(sc.transform(Fte))


def _tune_lr(sets, dr, cr, y, folds):
    """choose (key, C) by CV on the discovery rows, refit, scores on the confirmation rows.
    sets: {key: matrix over ALL rows}"""
    yd = y[dr]
    paths = {k: ts._cv_path([(F[dr][a], F[dr][b]) for a, b in folds], yd, folds) for k, F in sets.items()}
    k, c, s = ts._pick(paths)
    F = sets[k]
    return k, c, s, _fit_scores(F[dr], yd, F[cr], c)


def _tune_classical(Zall, C_all, dr, cr, y, folds, p):
    """classical_best on confirmation: curves + suite maps, chosen by discovery CV"""
    yd, Zd, Zc = y[dr], Zall[dr], Zall[cr]
    paths, final = {}, {}
    for key, F in C_all.items():
        paths[("curve", key)] = ts._cv_path([(F[dr][a], F[dr][b]) for a, b in folds], yd, folds)
        final[("curve", key)] = (F[dr], F[cr])
    for nm, D, fn in ts.classical_candidates(p, "v2sym"):
        try:
            pairs = [fn(Zd[a], yd[a], Zd[b]) for a, b in folds]
            o = fn(Zd, yd, Zc)
        except Exception:
            continue
        if o is None or any(q is None for q in pairs):
            continue
        arrs = [o[0], o[1]] + [x for q in pairs for x in q]
        if not all(np.isfinite(x).all() and np.abs(x).max() < 1e10 for x in arrs):
            continue
        paths[("suite", nm)] = ts._cv_path(pairs, yd, folds)
        final[("suite", nm)] = o
    k, c, s = ts._pick(paths)
    Ftr, Fte = final[k]
    return k, c, s, _fit_scores(Ftr, yd, Fte, c)


def _tune_nonlinear(Zall, dr, cr, y, folds):
    grid = ad.nl_grid(Zall.shape[1])
    yd = y[dr]
    sc = {(l, c): np.mean([ad._fit_auc(mk, Zall[dr][a], yd[a], Zall[dr][b], yd[b]) for a, b in folds])
          for l, c, mk in grid}
    (l, c) = max(sc, key=sc.get)
    mk = {(a, b): m for a, b, m in grid}[(l, c)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mk().fit(Zall[dr], yd)
    return f"{l}:{c}", sc[(l, c)], ad._score(m, Zall[cr])


def _confirm_job(args):
    out, name = args
    o = out / name
    if (o / "confirmation.npz").exists():
        return name, "skip"
    Z, Xall, y, disc, groups, info = load_prepared(out, name)
    p = Z.shape[1]
    dr, cr = np.flatnonzero(disc), np.flatnonzero(~disc)
    folds = outer_folds(y[dr], None if groups is None else groups[dr])
    t0 = time.time()
    scores, meta = {}, {}
    S = student_sets(out, name, p, slice(None))                # all rows; _tune_lr slices
    for arm, sets in S.items():
        k, c, s, sc = _tune_lr(sets, dr, cr, y, folds)            # lazy: one matrix at a time
        scores[arm], meta[arm] = sc, dict(choice=k, C=c, cv=s, D=family_size(k, p))
    C_all = ts.curve_sets(Z, D_STUDENTS)
    k, c, s, sc = _tune_classical(Z, C_all, dr, cr, y, folds, p)
    scores["classical_best"], meta["classical_best"] = sc, dict(choice=str(k), C=c, cv=s)
    for tag, M in (("", Z), ("ref_all_", Xall)):
        ch, s, sc = _tune_nonlinear(M, dr, cr, y, folds)
        scores[f"{tag}nonlinear_best"], meta[f"{tag}nonlinear_best"] = sc, dict(choice=ch, cv=s)
    # equal-count comparison: best classical expansion with D <= each map arm's own D
    for arm in ("S_Q", "S_MF"):
        DQ = meta[arm]["D"]
        sub = {key: F for key, F in C_all.items() if key[1] <= DQ}
        if sub:
            k, c, s, sc = _tune_lr(sub, dr, cr, y, folds)
            scores[f"curve_le_{arm}"], meta[f"curve_le_{arm}"] = sc, dict(choice=f"{k[0]}{k[1]}", C=c, cv=s, D=k[1])
    # D* on confirmation (descriptive): the discovery D* expansion, tuned and scored
    Cr = pd.read_csv(o / "discovery_curve.csv")
    R = pd.read_csv(o / "discovery_students.csv")
    target = R[R.arm == "S_Q"].auc.mean()
    cm = Cr.groupby(["family", "D"]).auc.mean().reset_index()
    ok = cm[cm.auc >= target - DESIGN["verdict_tol"]].sort_values("D")
    if len(ok):
        key = (ok.iloc[0].family, int(ok.iloc[0].D))
        k, c, s, sc = _tune_lr({key: C_all[key]}, dr, cr, y, folds)
        scores["curve_dstar"], meta["curve_dstar"] = sc, dict(choice=f"{key[0]}{key[1]}", C=c, cv=s, D=key[1])
    np.savez_compressed(o / "confirmation.npz", y=y[cr], groups=(groups[cr] if groups is not None else np.arange(len(cr))),
                        **{f"score__{a}": v for a, v in scores.items()})
    (o / "confirmation_choices.json").write_text(json.dumps(meta, indent=2, default=str))
    return name, f"ok {time.time() - t0:.0f}s"


def do_confirm(out, workers, only):
    names = [n for n in ready(out, only) if (out / n / "discovery_students.csv").exists()]
    return ad._pool(_confirm_job, [(out, n) for n in names], workers, "confirmation")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def verdict(m, tol=DESIGN["verdict_tol"], win=DESIGN["win_margin"]):
    """m: mean discovery AUC per arm (keys auc_*).  Exactly one verdict, in plan order."""
    q = m["auc_S_Q"]
    if max(m[c] for c in CLASSICAL) >= q - tol:
        return "classical enough"
    if max(m[c] for c in STANDINS) >= q - tol:
        return "mean-field enough"
    if q - max(m[c] for c in CLASSICAL + STANDINS) >= win:
        return "exact ahead (candidate)"
    return "exact slightly ahead"


def boot_diff(y, groups, s_a, s_bs, n=DESIGN["n_boot"], seed=0):
    """observed AUC(a) - max_b AUC(b) and its 95% interval, resampling GROUPS"""
    f = lambda idx: roc_auc_score(y[idx], s_a[idx]) - max(roc_auc_score(y[idx], s[idx]) for s in s_bs)
    obs = f(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    G = pd.Series(np.arange(len(y))).groupby(groups).apply(lambda s: s.to_numpy()).tolist()
    vals = []
    for _ in range(n):
        idx = np.concatenate([G[i] for i in rng.integers(0, len(G), len(G))])
        if len(np.unique(y[idx])) == 2:
            vals.append(f(idx))
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score_dataset(out, name):
    o = out / name
    info = json.loads((o / "PREP.json").read_text())
    row = dict(name=name, n=info["n"], positives=info["positives"], p=info.get("p"), status=info["status"],
               reason=info.get("reason", ""))
    if info["status"] != "ok" or not (o / "discovery_students.csv").exists():
        return row
    p = info["p"]
    R = pd.read_csv(o / "discovery_students.csv")
    N = pd.read_csv(o / "discovery_nonlinear.csv")
    m = {f"auc_{a}": v for a, v in pd.concat([R, N]).groupby("arm").auc.mean().items()}
    row.update(m)
    row["verdict"] = verdict(m)
    row["margin_disc"] = m["auc_S_Q"] - max(m[c] for c in CLASSICAL + STANDINS)
    for arm in ("S_Q", "S_MF"):
        row[f"D_{arm}"] = int(R[R.arm == arm].choice.map(lambda k: family_size(k, p)).mode().iloc[0])
    Cr = pd.read_csv(o / "discovery_curve.csv")
    cm = Cr.groupby(["family", "D"]).auc.mean().reset_index()
    row["Dstar"] = ts._dstar(cm, m["auc_S_Q"], DESIGN["verdict_tol"])
    row["R"] = row["Dstar"] / row["D_S_Q"]
    for arm in ("S_Q", "S_MF"):
        le = cm[cm.D <= row[f"D_{arm}"]].auc
        row[f"G_{arm}_disc"] = m[f"auc_{arm}"] - (le.max() if len(le) else np.nan)
    Ms = pd.read_csv(o / "discovery_measures.csv")
    row["F1_order"] = Ms.F1_order.iloc[0]
    row["L1_slope"] = Ms.L1_slope.iloc[0]
    if (o / "confirmation.npz").exists():
        Cf = np.load(o / "confirmation.npz")
        y, g = Cf["y"], Cf["groups"]
        s = {k.split("__", 1)[1]: Cf[k] for k in Cf.files if k.startswith("score__")}
        for a, v in s.items():
            row[f"conf_auc_{a}"] = roc_auc_score(y, v)
        others = [s[a] for a in ("classical_best", "nonlinear_best", "S_MF", "S_J0", "S_GS")]
        row["margin_conf"], row["margin_conf_lo"], row["margin_conf_hi"] = boot_diff(y, g, s["S_Q"], others)
        for arm in ("S_Q", "S_MF"):
            if f"curve_le_{arm}" in s:
                row[f"G_{arm}_conf"], row[f"G_{arm}_conf_lo"], _ = boot_diff(y, g, s[arm], [s[f"curve_le_{arm}"]], seed=1)
        row["Q_minus_MF_conf"], row["Q_minus_MF_conf_lo"], _ = boot_diff(y, g, s["S_Q"], [s["S_MF"]], seed=2)
        row["confirmed_win"] = bool(row["verdict"] == "exact ahead (candidate)"
                                    and row["margin_conf"] >= DESIGN["win_margin"] and row["margin_conf_lo"] > 0)
        gs = [(row.get(f"G_{a}_conf", np.nan), row.get(f"G_{a}_conf_lo", np.nan)) for a in ("S_Q", "S_MF")]
        row["confirmed_compact"] = any(gv >= DESIGN["compact_gap"] and lo > 0 for gv, lo in gs)
        row["confirmed_quantum_part"] = bool(row["confirmed_compact"] and row["Q_minus_MF_conf_lo"] > 0)
    return row


def checks(T):
    run = T[T.get("verdict").notna()] if "verdict" in T else T.iloc[0:0]
    conf = run[run.get("confirmed_win").notna()] if "confirmed_win" in run else run.iloc[0:0]
    n = len(run)
    rows = [
        ("R1", "verdict 'classical enough'", int((run.verdict == "classical enough").sum()) if n else 0, n,
         "prediction: on >= 5 of the datasets run" if n >= 5 else f"prediction: on >= 5 (only {n} run so far)"),
        ("R2", "confirmed exact wins", int(conf.confirmed_win.sum()) if len(conf) else 0, len(conf), "prediction: 0"),
        ("R3", "confirmed map compactness", int(conf.confirmed_compact.sum()) if len(conf) else 0, len(conf), "prediction: <= 1"),
        ("R4", "confirmed quantum part (among compact)", int(conf.confirmed_quantum_part.sum()) if len(conf) else 0,
         int(conf.confirmed_compact.sum()) if len(conf) else 0, "prediction: 0"),
        ("R5", "F1 order 1 (additive)", int((run.F1_order.astype(str) == "1").sum()) if n else 0, n,
         "prediction: on >= 5 of the datasets run"),
    ]
    return pd.DataFrame(rows, columns=["check", "what", "count", "of", "prediction"])


def checks_pool(T):
    """Amendment 3 predictions for the 55-dataset pool (written knowing Run 2's results)."""
    run = T[T.verdict.notna()] if "verdict" in T else T.iloc[0:0]
    conf = run[run.confirmed_win.notna()] if "confirmed_win" in run else run.iloc[0:0]
    n, nc = len(run), len(conf)
    wins = list(conf.name[conf.confirmed_win.astype(bool)]) if nc else []
    qp = list(conf.name[conf.confirmed_quantum_part.astype(bool)]) if nc else []
    j0 = int(((run.auc_S_Q - run.auc_S_J0) <= DESIGN["verdict_tol"]).sum()) if n else 0
    rows = [
        ("Q1", "verdict 'classical enough'", int((run.verdict == "classical enough").sum()) if n else 0, n,
         "prediction: on >= 90% of the datasets run"),
        ("Q2", "confirmed exact wins", len(wins), nc,
         "prediction: <= 1, and only monk2 or heart" + (f" (got: {', '.join(wins)})" if wins else "")),
        ("Q3", "confirmed map compactness", int(conf.confirmed_compact.sum()) if nc else 0, nc, "prediction: <= 2"),
        ("Q4", "confirmed quantum part", len(qp), int(conf.confirmed_compact.sum()) if nc else 0,
         "prediction: 0, except possibly monk2" + (f" (got: {', '.join(qp)})" if qp else "")),
        ("Q5", "no-coupling (J=0) within 0.01 of the exact map", j0, n, "prediction: on >= 80% of the datasets run"),
    ]
    K = pd.DataFrame(rows, columns=["check", "what", "count", "of", "prediction"])
    if n and "boolean_rule" in run:
        for lab, g in (("Boolean-rule datasets", run[run.boolean_rule]), ("all other datasets", run[~run.boolean_rule])):
            K.loc[len(K)] = ("Q6", f"{lab}: interaction ladder order >= 3",
                             int(g.F1_order.astype(str).isin(["3", ">=4"]).sum()), len(g), "reported")
            K.loc[len(K)] = ("Q6", f"{lab}: median discovery margin (x1000)",
                             int(round(1000 * g.margin_disc.median())) if len(g) else 0, len(g), "reported")
    return K


def do_score(out, only, group="ophtho"):
    names = [n for n in only if (out / n / "PREP.json").exists()]
    T = pd.DataFrame([score_dataset(out, n) for n in names])
    if group == "pool":
        T["boolean_rule"] = T.name.isin(RD.POOL_BOOLEAN)
    T.to_csv(out / "REAL_DATASETS.csv", index=False)
    K = checks(T) if group == "ophtho" else checks_pool(T)
    K.to_csv(out / "REAL_RESULTS.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = [c for c in ["name", "n", "positives", "p", "status", "verdict", "auc_S_Q", "auc_S_MF", "auc_classical_best",
                        "auc_nonlinear_best", "auc_ref_all_nonlinear_best", "margin_disc", "D_S_Q", "Dstar", "R",
                        "G_S_Q_disc", "margin_conf", "margin_conf_lo", "G_S_Q_conf", "G_S_Q_conf_lo",
                        "Q_minus_MF_conf", "confirmed_win", "confirmed_compact", "F1_order", "reason"] if c in T]
    print(T[cols].round(3).to_string(index=False))
    print()
    print(K.to_string(index=False))
    print(f"\nwrote {out / 'REAL_DATASETS.csv'} and {out / 'REAL_RESULTS.csv'}")


def do_status(only):
    print(f"data folder: {RD.DATA_ROOT}")
    for name in only:
        try:
            r = RD.load(name)
            print(f"  [ready]     {name}: {len(r.y)} rows, {r.X.shape[1]} candidate features, "
                  f"{int(r.y.sum())} positive" + (", grouped" if r.groups is not None else ""))
        except RD.NotReady as e:
            print(f"  [not ready] {name}: {str(e).splitlines()[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["status", "prepare", "banks", "run", "confirm", "score"])
    ap.add_argument("--group", choices=["ophtho", "pool"], default="ophtho",
                    help="ophtho: the ophthalmic datasets (default); pool: the 55 Run-2 datasets (Amendment 3)")
    ap.add_argument("--out", type=Path, default=None, help="default runs/realdata (ophtho) or runs/realdata_pool")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.out = a.out or Path("runs/realdata" if a.group == "ophtho" else "runs/realdata_pool")
    a.only = a.only or DESIGN["datasets" if a.group == "ophtho" else "pool"]
    if a.phase == "status":
        do_status(a.only)
    elif a.phase == "prepare":
        do_prepare(a.out, a.only)
    elif a.phase == "banks":
        sys.exit(1 if do_banks(a.out, a.workers, a.only) else 0)
    elif a.phase == "run":
        sys.exit(1 if do_run(a.out, a.workers, a.only) else 0)
    elif a.phase == "confirm":
        sys.exit(1 if do_confirm(a.out, a.workers, a.only) else 0)
    else:
        do_score(a.out, a.only, a.group)


if __name__ == "__main__":
    main()
