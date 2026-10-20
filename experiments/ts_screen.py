#!/usr/bin/env python3
"""
Screen study (docs/screen_plan.md, with Amendments 1-3): can cheap tests on (x, y)
say "this dataset resists feature engineering, and not just because the labels
are noisy", and how often does AQFM help on datasets like that?

This file builds the two new blocks of the library and runs the students on them.
The screen measures (L1, L2, L3, F1) and the checks (P1-P12) come in a later
phase, after the labels have been looked at.

Two new blocks, all on the SAME shared inputs as the tau sweep (one X per p), so
every student feature bank already exists and nothing is re-simulated:

  aqfm_noise  126 noisier copies (temperature T = 1.0) of tau-sweep teachers:
              exact and mean-field, local readout, kappa in {0.5, 2, 8},
              tau in {1, 3, 5, 10, 20, 40}, p in {4, 6, 8}; plus J=0, seed 0.
              Each copy uses the ORIGINAL dataset's seed, so the same random
              weights w and the same random draws: only the temperature differs
              (paired with the T = 0.25 original in runs/tau_sweep).
  foreign     162 non-AQFM teachers: 9 classically defined score functions s(z)
              x p in {4, 6, 8} x 3 seeds x T in {0.25, 1.0} (also paired in T).

Labels, for both blocks, exactly as in the tau sweep:
    score = standardize(s(z));   y ~ Bernoulli( sigmoid(score / T) )

Phases
  make   labels for both blocks -> OUT/labels/, OUT/SCREEN_TEACHERS.csv,
         OUT/SCREEN_PROTOCOL.json.  OUT/banks is a link to the tau sweep's banks.
  run    every student of the tau sweep (same menus, folds, tuning) and the
         nonlinear add-on (HGB / RBF SVM / MLP) on every new dataset.  Restartable.
  look   quick sanity table per teacher family: true ceiling, positive rate,
         and three fast classical models (no map).  Needs only the inputs, so it
         can run before `run`.
  measures  the screen measures (L1, L2, L3, F1; experiments/screen_measures.py)
         on EVERY library dataset: the 864 tau-sweep datasets (labels read from
         --sweep) and the 288 new ones.  Needs only inputs and labels, so it can run
         alongside `run`.  -> OUT/measures/<name>.csv, OUT/SCREEN_MEASURES.csv

Usage (from the repo root):
  python experiments/ts_screen.py make --out runs/screen --sweep runs/tau_sweep
  python experiments/ts_screen.py look --out runs/screen
  python experiments/ts_screen.py run  --out runs/screen --workers 32
  python experiments/ts_screen.py measures --out runs/screen --sweep runs/tau_sweep --workers 32
Nothing in the tau-sweep folder is ever written.
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
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_student as ts            # noqa: E402
import ts_tau_sweep as tt               # noqa: E402
import ts_addons as ad                  # noqa: E402

TEACHERS_FILE = "SCREEN_TEACHERS.csv"

DESIGN = {
    "version": "screen_1",
    "plan": "docs/screen_plan.md (Amendments 1-3)",
    "n": tt.DESIGN["n"],
    "ps": [4, 6, 8],
    # aqfm_noise block: which tau-sweep teachers get a T = 1.0 copy
    "noise_temp": 1.0,
    "noise_kinds": ["Q", "MF"],
    "noise_kappas": [0.5, 2.0, 8.0],
    "noise_taus": [1.0, 3.0, 5.0, 10.0, 20.0, 40.0],
    "noise_readout": "local",
    "noise_J0_seed": 0,
    # foreign block
    "foreign_families": ["LIN", "PAIR", "TRIPLE", "MIXED", "PAR3", "PARp", "NET1", "NETD", "NOISE"],
    "foreign_seeds": 3,
    "foreign_temps": [0.25, 1.0],
    "net_width": 16,
    "netd_gain": 3.0,                 # Amendment 2
    "parity_sharpness": 2.0,          # tanh(2 z): close to sign(z) but smooth
}


# ---------------------------------------------------------------------------
# foreign (non-AQFM) teachers
# ---------------------------------------------------------------------------
# Every family is a score function s(Z) of the standardized, clipped inputs Z
# (shape n x p).  Its random pieces (weights, which inputs interact) are drawn
# from `rng` first; the label draws come after, from the same rng.  Because the
# structure draws come first and are the same for every temperature, the T = 0.25
# and T = 1.0 copies share the same s(z) AND the same uniform random numbers.

def _pick_subsets(rng, p, k, how_many):
    """`how_many` distinct k-subsets of the p inputs (all of them if fewer exist)."""
    allsub = list(combinations(range(p), k))
    m = min(how_many, len(allsub))
    idx = rng.choice(len(allsub), size=m, replace=False)
    return [allsub[i] for i in sorted(idx)]


def _products(Z, subsets, a):
    """sum over subsets S of a_S * prod_{i in S} z_i"""
    return sum(ai * np.prod(Z[:, list(S)], axis=1) for ai, S in zip(a, subsets))


def foreign_score(family, Z, rng, d=DESIGN):
    """Raw (unstandardized) score s(Z) for one foreign teacher, plus a short
    description of the structure that was drawn (saved in the teacher table)."""
    n, p = Z.shape
    if family == "LIN":
        w = rng.standard_normal(p)
        return Z @ w, "linear"
    if family == "PAIR":
        S = _pick_subsets(rng, p, 2, p)
        a = rng.standard_normal(len(S))
        return _products(Z, S, a), f"pairs={S}"
    if family == "TRIPLE":
        S = _pick_subsets(rng, p, 3, p)
        a = rng.standard_normal(len(S))
        return _products(Z, S, a), f"triples={S}"
    if family == "MIXED":
        # three parts of different interaction order, each scaled to unit variance,
        # so no single order dominates
        w = rng.standard_normal(p)
        S2 = _pick_subsets(rng, p, 2, p)
        a2 = rng.standard_normal(len(S2))
        S3 = _pick_subsets(rng, p, 3, p)
        a3 = rng.standard_normal(len(S3))
        parts = [Z @ w, _products(Z, S2, a2), _products(Z, S3, a3)]
        return sum(ts._std(v) for v in parts), f"pairs={S2};triples={S3}"
    if family in ("PAR3", "PARp"):
        # parity-like: the product of tanh(2 z_i) over a subset is close to the
        # product of the SIGNS of those inputs, i.e. "an even or odd number of them
        # are negative".  The classic case where no subset smaller than the whole
        # carries any signal.
        k = 3 if family == "PAR3" else p
        S = list(_pick_subsets(rng, p, k, 1)[0])
        return np.prod(np.tanh(d["parity_sharpness"] * Z[:, S]), axis=1), f"inputs={S}"
    if family == "NET1":
        # small random one-hidden-layer network.  With these weight sizes it is
        # close to linear: an EASY nonlinear control
        h = d["net_width"]
        W = rng.standard_normal((h, p)) / np.sqrt(p)
        b = rng.standard_normal(h)
        v = rng.standard_normal(h)
        return np.tanh(Z @ W.T + b) @ v, f"width={h};layers=1;gain=1"
    if family == "NETD":
        # harder random network (Amendment 2): two hidden layers, weights 3x
        # larger (sharper switching), and its LINEAR part removed (least squares on
        # these inputs), so a linear model has nothing to ride on
        h, g = d["net_width"], d["netd_gain"]
        A = Z
        for _ in range(2):
            W = rng.standard_normal((h, A.shape[1])) * g / np.sqrt(A.shape[1])
            b = rng.standard_normal(h)
            A = np.tanh(A @ W.T + b)
        s = A @ rng.standard_normal(h)
        X1 = np.column_stack([np.ones(n), Z])
        s = s - X1 @ np.linalg.lstsq(X1, s, rcond=None)[0]
        return s, f"width={h};layers=2;gain={g:g};linear part removed"
    if family == "NOISE":
        # independent of Z: nothing to learn (true ceiling 0.5)
        return rng.standard_normal(n), "independent of inputs"
    raise ValueError(family)


def foreign_specs(d=DESIGN):
    specs = []
    for fam, p, s, T in product(d["foreign_families"], d["ps"], range(d["foreign_seeds"]), d["foreign_temps"]):
        base = f"fr_{fam}_p{p}_s{s}"                       # seed comes from the name WITHOUT the temperature
        specs.append(dict(block="foreign", kind=fam, p=p, kappa=np.nan, tau=np.nan, readout="",
                          seed=s, temp=T, base=base, name=f"{base}_T{T:g}", data_seed=tt._seed(base)))
    return specs


def make_foreign(sp, d=DESIGN):
    """labels for one foreign teacher -> (y, standardized score, meta)"""
    _, Z = tt.inputs(sp["p"])
    rng = np.random.default_rng(sp["data_seed"])
    raw, structure = foreign_score(sp["kind"], Z, rng, d)
    score = ts._std(raw)
    prob = 1.0 / (1.0 + np.exp(-score / sp["temp"]))
    y = (rng.random(d["n"]) < prob).astype(int)
    # true ceiling = best AUC any model of z could reach.  For a deterministic s(z)
    # the best model is s itself; for NOISE, s does not depend on z, so it is 0.5.
    oracle = roc_auc_score(y, score)
    ceiling = 0.5 if sp["kind"] == "NOISE" else oracle
    meta = dict(sp, structure=structure, oracle_auc=oracle, ceiling=ceiling,
                pos_rate=float(y.mean()), degenerate=False)
    return y, score, meta


# ---------------------------------------------------------------------------
# noisier copies of tau-sweep teachers
# ---------------------------------------------------------------------------

def noise_specs(d=DESIGN):
    """the 126 tau-sweep teachers that get a T = 1.0 copy"""
    keep = []
    for sp in tt.teacher_specs():
        if sp["readout"] != d["noise_readout"] or sp["p"] not in d["ps"] or sp["tau"] not in d["noise_taus"]:
            continue
        if sp["kind"] in d["noise_kinds"] and sp["kappa"] in d["noise_kappas"]:
            keep.append(sp)
        elif sp["kind"] == "J0" and sp["seed"] == d["noise_J0_seed"]:
            keep.append(sp)
    out = []
    for sp in keep:
        # data_seed stays the ORIGINAL one (tt.teacher_specs derives it from the
        # original name), so w and the random draws are identical to the original
        out.append(dict(sp, block="aqfm_noise", temp=d["noise_temp"], base=sp["name"],
                        name=f"{sp['name']}_T{d['noise_temp']:g}"))
    return out


def make_noise_copy(out, sp, d=DESIGN):
    """labels for one T = 1.0 copy, built by the tau sweep's own make_labels with
    only the temperature changed"""
    y, meta = tt.make_labels(out, sp, dict(tt.DESIGN, temp=sp["temp"]))
    if y is not None:
        meta["ceiling"] = meta["oracle_auc"]          # deterministic score -> oracle is the ceiling
    return y, meta


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def _link_banks(out, sweep):
    """OUT/banks -> the tau sweep's banks (read only; never written here)"""
    src = (sweep / "banks").resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"no banks in {sweep}")
    dst = out / "banks"
    if dst.is_symlink() or dst.exists():
        if dst.resolve() != src:
            raise RuntimeError(f"{dst} exists and does not point to {src}")
        return
    dst.symlink_to(src)


def do_make(out, sweep, d=DESIGN):
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    proto = {"design": d, "student_design": tt.DESIGN, "sweep": str(sweep.resolve()),
             "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "note": "written by `make`, before any student was run on these datasets"}
    (out / "SCREEN_PROTOCOL.json").write_text(json.dumps(proto, indent=2, default=str))
    metas = []
    for sp in foreign_specs(d):
        y, _, meta = make_foreign(sp, d)
        np.save(out / "labels" / f"{sp['name']}.npy", y)
        metas.append(meta)
    _link_banks(out, sweep)
    for sp in noise_specs(d):
        y, meta = make_noise_copy(out, sp, d)
        if y is not None:
            np.save(out / "labels" / f"{sp['name']}.npy", y)
        metas.append(meta)
    M = pd.DataFrame(metas)
    M.to_csv(out / TEACHERS_FILE, index=False)
    print(f"made {int((~M.degenerate.astype(bool)).sum())} datasets "
          f"({int(M.degenerate.astype(bool).sum())} degenerate) -> {out / TEACHERS_FILE}")
    print(M.groupby(["block", "kind", "temp"])[["ceiling", "pos_rate"]].median().round(3).to_string())


def do_run(out, workers):
    M = pd.read_csv(out / TEACHERS_FILE)
    M = M[~M.degenerate.astype(bool)]
    # 1. the tau sweep's students (same menus, folds, C tuning, classical curves)
    jobs = sorted([(out, n, int(p)) for n, p in zip(M.name, M.p)], key=lambda j: (-j[2], j[1]))
    fails = ad._pool(tt._job, jobs, workers, "student")
    # 2. the nonlinear learners (same outer folds)
    fails += ad.do_nonlinear(out, workers, TEACHERS_FILE)
    sys.exit(1 if fails else 0)


def _quick(Z, y):
    """three fast classical models, 5-fold CV AUC (a sanity look, NOT the students)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "lin": LogisticRegression(max_iter=2000),
        "poly3": make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(),
                               LogisticRegression(C=0.1, max_iter=5000)),
        "hgb": HistGradientBoostingClassifier(max_iter=200, random_state=0),
    }
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, m in models.items():
            s = cross_val_predict(m, Z, y, cv=cv, method="predict_proba")[:, 1]
            res[f"quick_{k}"] = roc_auc_score(y, s)
    return res


def do_look(out, d=DESIGN):
    M = pd.read_csv(out / TEACHERS_FILE)
    M = M[(M.block == "foreign") & ~M.degenerate.astype(bool)]
    rows = []
    for r in M.itertuples():
        _, Z = tt.inputs(int(r.p))
        y = np.load(out / "labels" / f"{r.name}.npy")
        rows.append(dict(name=r.name, kind=r.kind, p=r.p, temp=r.temp, ceiling=r.ceiling,
                         pos_rate=r.pos_rate, **_quick(Z, y)))
    L = pd.DataFrame(rows)
    L.to_csv(out / "SCREEN_LOOK.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["ceiling", "pos_rate", "quick_lin", "quick_poly3", "quick_hgb"]
    order = {f: i for i, f in enumerate(d["foreign_families"])}
    for T in d["foreign_temps"]:
        G = L[L.temp == T].groupby(["kind", "p"])[cols].median().round(3).reset_index()
        G = G.sort_values(["kind", "p"], key=lambda c: c.map(order) if c.name == "kind" else c)
        print(f"\n=== foreign teachers, T = {T:g}: medians over 3 seeds ===")
        print(G.to_string(index=False))
    print(f"\nwrote {out / 'SCREEN_LOOK.csv'}")


def _measure_job(args):
    """one dataset's screen measures.  Only the inputs and the labels are loaded:
    screen_measures never sees banks, teacher scores or ceilings."""
    import screen_measures as SM
    out, name, p, labels = args
    f = out / "measures" / f"{name}.csv"
    if f.exists():
        return name, "skip"
    _, Z = tt.inputs(p)
    y = np.load(labels)
    t0 = time.time()
    row = SM.all_measures(Z, y, name)
    row["seconds"] = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.csv")
    pd.DataFrame([row]).to_csv(tmp, index=False)
    tmp.replace(f)
    return name, f"ok {row['seconds']:.0f}s"


def library(out, sweep):
    """every library dataset: (name, p, label file, block)"""
    T = pd.read_csv(sweep / "TT_TEACHERS.csv")
    T = T[~T.degenerate.astype(bool)]
    S = pd.read_csv(out / TEACHERS_FILE)
    S = S[~S.degenerate.astype(bool)]
    rows = [(n, int(p), sweep / "labels" / f"{n}.npy", "tau_sweep") for n, p in zip(T.name, T.p)]
    rows += [(n, int(p), out / "labels" / f"{n}.npy", b) for n, p, b in zip(S.name, S.p, S.block)]
    return rows


def do_measures(out, sweep, workers):
    L = library(out, sweep)
    jobs = sorted([(out, n, p, lab) for n, p, lab, _ in L], key=lambda j: (-j[2], j[1]))
    fails = ad._pool(_measure_job, jobs, workers, "measure")
    parts = [pd.read_csv(out / "measures" / f"{n}.csv") for n, *_ in L if (out / "measures" / f"{n}.csv").exists()]
    if parts:
        M = pd.concat(parts, ignore_index=True)
        M = M.merge(pd.DataFrame([(n, b) for n, _, _, b in L], columns=["name", "block"]), on="name")
        M.to_csv(out / "SCREEN_MEASURES.csv", index=False)
        print(f"wrote {out / 'SCREEN_MEASURES.csv'} ({len(M)} of {len(L)} datasets)")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["make", "run", "look", "measures"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, default=Path("runs/tau_sweep"),
                    help="tau-sweep run folder (banks are read from here)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "make":
        do_make(a.out, a.sweep)
    elif a.phase == "run":
        do_run(a.out, a.workers)
    elif a.phase == "measures":
        do_measures(a.out, a.sweep, a.workers)
    else:
        do_look(a.out)


if __name__ == "__main__":
    main()
