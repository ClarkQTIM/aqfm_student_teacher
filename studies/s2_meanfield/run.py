#!/usr/bin/env python3
"""
S2: does mean-field catch up with the exact map as the number of qubits p grows?
Plan and predictions: studies/s2_meanfield/PLAN.md (fixed before any run).

No labels to learn and no learners: we simulate the same 400 inputs with the exact
map and with mean-field, at every (p, cell), and measure how far apart they are.

Phases
  time     time a few exact rows at one p and estimate the cost of `banks` there
           (used to decide whether p = 16 is affordable)
  banks    simulate block 0 (pool rows 0-399) for Q and MF at every (p, cell)
             -> runs/studies/banks/alltoall/p{p}/{kind}_c{c}_t{tau}/block0000.npy
  measure  per (p, cell): the per-sample numbers behind E, C_pair, C_ZZ, C_coll, U
             -> runs/studies/s2/persample/p{p}_{cell}.npz, runs/studies/s2/S2_MEASURES.csv
  score    exponents alpha (with bootstrap intervals) and the checks M1-M5
             -> runs/studies/s2/S2_ALPHAS.csv, S2_CHECKS.csv
  amend    post-hoc Amendment 1 (PLAN.md): U on 800 rows (A2) and against the best of the
           60 mean-field menu settings (A1); needs `banks --blocks 2 --menu` first
             -> runs/studies/s2/S2_AMEND.csv, S2_AMEND_ALPHAS.csv, S2_AMEND_CHECKS.csv

Usage (from the repo root; see the chat for the full copy-paste block):
  python studies/s2_meanfield/run.py time    --p 16
  python studies/s2_meanfield/run.py banks   --ps 2,3,4,5,6,7,8,10,12,14 --workers 31
  python studies/s2_meanfield/run.py measure --workers 31
  python studies/s2_meanfield/run.py score
  python studies/s2_meanfield/run.py banks   --ps 2,3,4,5,6,7,8,10,12,14 --blocks 2 --menu --workers 31
  python studies/s2_meanfield/run.py amend   --workers 31
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: workers x 1 <= 32 cores

import argparse
import hashlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, slices_for_p, strang_steps   # noqa: E402
from studies.common import bankstore as bs                                   # noqa: E402
from studies.common.cells import cell_name, cells, coupling                  # noqa: E402

DESIGN = {
    "version": "s2_1",
    "plan": "studies/s2_meanfield/PLAN.md",
    "ps": [2, 3, 4, 5, 6, 7, 8, 10, 12, 14],     # 16 added by a second `banks` call if `time` allows
    "fit_p_min": 4,                              # exponents are fitted on p >= 4
    "n_weights": 20,                             # readout weight vectors per cell for U
    "cv_folds": 5,
    "n_boot": 1000,
    "u_target": 0.01,                            # M5: p at which U would fall below 1%
    "m2_min_cells": 9,                           # M2: at least 9 of 12 window cells
}
# Amendment 1: the tau sweep's mean-field student menu (5 couplings x 12 anneal times)
MENU = [(c, t) for c in (0.5, 1.0, 2.0, 4.0, 8.0)
        for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0)]
AMEND_BLOCKS = 2                                 # Amendment 2: 800 rows instead of 400
BANKS = Path("runs/studies/banks/alltoall")
OUT = Path("runs/studies/s2")
MEASURES = ["E", "C_pair", "C_ZZ", "C_coll", "U"]


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

def do_time(p: int, workers: int, rows: int = 4):
    """time `rows` exact rows at tau = 20 and scale by step count to every cell"""
    Z = bs.pool_inputs(p)[:rows]
    t0 = time.time()
    compute_bank(Z, coupling(p, 4.0, "Q"), 20.0, method="strang")
    per_row = (time.time() - t0) / rows
    per_step = per_row / strang_steps(20.0)
    total = sum(per_step * strang_steps(cl["tau"]) * bs.BLOCK for cl in cells())
    print(f"p = {p}: {per_row:.1f} s per row at tau = 20 (one core)")
    print(f"estimated `banks` at p = {p} (14 cells x {bs.BLOCK} rows, exact only; mean-field is negligible): "
          f"{total / 3600:.1f} core-hours = about {total / workers / 60:.0f} min on {workers} workers")


# ---------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------

def bank_jobs(ps, blocks=1, menu=False):
    """pieces for the cells' exact and mean-field banks (blocks 0..blocks-1); with
    menu=True also every mean-field setting of the tau sweep's student menu (Amendment 1)"""
    specs = [(p, kind, cl["c"], cl["tau"]) for p in ps for cl in cells() for kind in ("Q", "MF")]
    if menu:
        specs += [(p, "MF", c, t) for p in ps for c, t in MENU]
    jobs = []
    for p, kind, c, t in dict.fromkeys(specs):           # unique, order kept
        for b in range(blocks):
            for s, e in bs.piece_specs(BANKS, p, kind, c, t, b):
                jobs.append((BANKS, p, kind, c, t, b, s, e))
    # slowest first: exact, large p, long anneals
    jobs.sort(key=lambda j: -((j[2] == "Q") * (2.0 ** j[1]) * strang_steps(j[4]) * (j[7] - j[6])))
    return jobs


def do_banks(ps, workers, blocks=1, menu=False):
    jobs = bank_jobs(ps, blocks, menu)
    _pool(bs.compute_piece, jobs, workers, f"bank pieces for p = {ps}, blocks 0-{blocks - 1}"
          + (", with the mean-field menu" if menu else ""))
    missing, n = [], 0
    for p, kind, c, t in dict.fromkeys((j[1], j[2], j[3], j[4]) for j in jobs):
        for b in range(blocks):
            n += 1
            if not bs.merge_block(BANKS, p, kind, c, t, b):
                missing.append(f"p{p} {kind} {cell_name(c, t)} block {b}")
    if missing:
        print(f"STOP: {len(missing)} bank blocks incomplete (rerun the same command to finish): {missing[:5]} ...")
        sys.exit(1)
    print(f"all {n} bank blocks complete")


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def _standardize(F):
    sd = F.std(0)
    keep = sd > 1e-9
    return (F[:, keep] - F[:, keep].mean(0)) / sd[keep]


def per_sample(BQ, BMF, p, c, cell, n_weights=None, folds=None):
    """The per-sample numbers behind each measure (so `score` can bootstrap rows).

    e[m]        mean over qubits of |r_Q - r_MF|^2, r = Bloch vector (<X>,<Y>,<Z>)
                  -> E = sqrt(mean_m e[m])
    cpair[m]    mean over pairs and P in {X,Y,Z} of |<P_i P_j> - <P_i><P_j>| (exact map)
    czz[m]      the same, Z only
    ccoll[m]    J * mean_i | sum_{j != i} C^ZZ_ij |  (signed sum: correlations that add up)
    r2[m,k], d2[m,k]
                for readout weights w_k: exact score s_k = standardized(F_Q w_k) (the tau
                sweep's teacher recipe, no coin flip); out-of-fold residual of a linear fit of
                s_k on the 3p mean-field features (5-fold), squared; and s_k's squared
                deviation.  U = median_k sum_m r2 / sum_m d2.
    """
    n_weights = n_weights or DESIGN["n_weights"]
    folds = folds or DESIGN["cv_folds"]
    sl, _ = slices_for_p(p)
    J = coupling(p, c, "Q")
    m = BQ.shape[0]
    rQ = np.stack([BQ[:, sl[k]] for k in "XYZ"], -1).astype(float)      # (m, p, 3)
    rM = np.stack([BMF[:, sl[k]] for k in "XYZ"], -1).astype(float)
    e = ((rQ - rM) ** 2).sum(-1).mean(1)

    pairs = list(combinations(range(p), 2))
    ii = np.array([a for a, _ in pairs]); jj = np.array([b for _, b in pairs])
    conn = {}
    for k in "XYZ":
        single = BQ[:, sl[k]].astype(float)
        conn[k] = BQ[:, sl[k + k]].astype(float) - single[:, ii] * single[:, jj]   # (m, q)
    cpair = np.mean([np.abs(conn[k]).mean(1) for k in "XYZ"], axis=0)
    czz = np.abs(conn["Z"]).mean(1)
    Cm = np.zeros((m, p, p))
    Cm[:, ii, jj] = conn["Z"]; Cm[:, jj, ii] = conn["Z"]
    ccoll = J * np.abs(Cm.sum(2)).mean(1)

    S = exact_scores(BQ, p, cell, n_weights)
    r2 = oof_residuals(S, mf_design(BMF, p), folds)
    d2 = (S - S.mean(0)) ** 2
    return dict(e=e, cpair=cpair, czz=czz, ccoll=ccoll, r2=r2, d2=d2)


def exact_scores(BQ, p, cell, n_weights=None):
    """(m, K) noise-free exact-teacher scores: random weights w_k on the standardized 3p
    exact readout, then standardized (the tau sweep's recipe without the coin flip).
    The weights depend only on (cell, p), so every analysis sees the same K teachers."""
    n_weights = n_weights or DESIGN["n_weights"]
    sl, _ = slices_for_p(p)
    FQ = _standardize(np.hstack([BQ[:, sl[k]] for k in "XYZ"]).astype(float))
    W = np.random.default_rng(_seed(f"s2_weights_{cell}_p{p}")).standard_normal((FQ.shape[1], n_weights))
    S = FQ @ W
    return (S - S.mean(0)) / S.std(0)


def mf_design(BMF, p):
    """intercept + the standardized 3p mean-field readout (<X>, <Y>, <Z> per qubit)"""
    sl, _ = slices_for_p(p)
    G = _standardize(np.hstack([BMF[:, sl[k]] for k in "XYZ"]).astype(float))
    return np.hstack([np.ones((len(G), 1)), G])


def oof_residuals(S, G, folds=None):
    """squared out-of-fold residuals of a least-squares fit of every column of S on G
    (k-fold, one fixed fold assignment that depends only on the number of rows)"""
    folds = folds or DESIGN["cv_folds"]
    m = len(S)
    order = np.random.default_rng(0).permutation(m)
    fold_of = np.empty(m, int)
    fold_of[order] = np.arange(m) % folds
    pred = np.empty_like(S)
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        coef, *_ = np.linalg.lstsq(G[tr], S[tr], rcond=None)
        pred[te] = G[te] @ coef
    return (S - pred) ** 2


def point_values(a):
    """the five measures from per-sample arrays (rows can be a bootstrap resample)"""
    return dict(E=float(np.sqrt(a["e"].mean())), C_pair=float(a["cpair"].mean()),
                C_ZZ=float(a["czz"].mean()), C_coll=float(a["ccoll"].mean()),
                U=float(np.median(a["r2"].sum(0) / a["d2"].sum(0))))


def measure_job(args):
    p, cl = args
    cell = cell_name(cl["c"], cl["tau"])
    f = OUT / "persample" / f"p{p}_{cell}.npz"
    if not f.exists():
        BQ = bs.load_rows(BANKS, p, "Q", cl["c"], cl["tau"])
        BMF = bs.load_rows(BANKS, p, "MF", cl["c"], cl["tau"])
        a = per_sample(BQ, BMF, p, cl["c"], cell)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp.npz")
        np.savez_compressed(tmp, **a)
        tmp.replace(f)
    a = dict(np.load(f))
    return dict(p=p, cell=cell, c=cl["c"], tau=cl["tau"], region=cl["region"], **point_values(a))


def available_ps():
    ps = []
    for d in sorted(BANKS.glob("p*")):
        p = int(d.name[1:])
        if all(bs.block_path(BANKS, p, k, cl["c"], cl["tau"], 0).exists() for cl in cells() for k in ("Q", "MF")):
            ps.append(p)
    return sorted(ps)


def do_measure(workers):
    ps = available_ps()
    if not ps:
        raise SystemExit(f"STOP: no complete banks under {BANKS}; run `banks` first")
    rows = []
    jobs = [(p, cl) for p in ps for cl in cells()]
    print(f"measuring {len(jobs)} (p, cell) pairs for p = {ps}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for fu in as_completed([ex.submit(measure_job, j) for j in jobs]):
            rows.append(fu.result())
    M = pd.DataFrame(rows).sort_values(["region", "c", "tau", "p"])
    M.to_csv(OUT / "S2_MEASURES.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nmedian over cells, by region and p:")
    print(M.groupby(["region", "p"])[MEASURES].median().to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {OUT / 'S2_MEASURES.csv'}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def fit_alpha(ps, vals):
    """log(v) = a - alpha * log(p), least squares.  Returns (alpha, a)."""
    x, y = np.log(np.asarray(ps, float)), np.log(np.maximum(np.asarray(vals, float), 1e-300))
    slope, a = np.polyfit(x, y, 1)
    return -slope, a


def bootstrap_values(ps, cell_list, n_boot):
    """For every p, n_boot resamples of the 400 rows, THE SAME resample for every cell
    (all cells share the same inputs at a given p).  Returns {(cell, measure): array
    (n_boot, len(ps))} of each measure recomputed on each resample."""
    out = {(cl, k): np.empty((n_boot, len(ps))) for cl in cell_list for k in MEASURES}
    for j, p in enumerate(ps):
        A = {cl: dict(np.load(OUT / "persample" / f"p{p}_{cl}.npz")) for cl in cell_list}
        m = len(next(iter(A.values()))["e"])
        idx = np.random.default_rng(_seed(f"s2_boot_p{p}")).integers(0, m, (n_boot, m))
        for cl, a in A.items():
            out[(cl, "E")][:, j] = np.sqrt(a["e"][idx].mean(1))
            out[(cl, "C_pair")][:, j] = a["cpair"][idx].mean(1)
            out[(cl, "C_ZZ")][:, j] = a["czz"][idx].mean(1)
            out[(cl, "C_coll")][:, j] = a["ccoll"][idx].mean(1)
            out[(cl, "U")][:, j] = np.median(a["r2"][idx].sum(1) / a["d2"][idx].sum(1), axis=1)
    return out


def _ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def do_score():
    M = pd.read_csv(OUT / "S2_MEASURES.csv")
    ps = sorted(p for p in M.p.unique() if p >= DESIGN["fit_p_min"])
    if len(ps) < 3:
        raise SystemExit(f"STOP: need at least 3 values of p >= {DESIGN['fit_p_min']}, have {ps}")
    info = {cell_name(cl["c"], cl["tau"]): cl for cl in cells()}
    cell_list = [c for c in info if set(ps) <= set(M[M.cell == c].p)]
    boot = bootstrap_values(ps, cell_list, DESIGN["n_boot"])
    rows, alpha_b = [], {}
    for cl in cell_list:
        sub = M[M.cell == cl].set_index("p")
        for k in MEASURES:
            al, a = fit_alpha(ps, sub.loc[ps, k])
            ab = np.array([fit_alpha(ps, boot[(cl, k)][b])[0] for b in range(DESIGN["n_boot"])])
            alpha_b[(cl, k)] = ab
            lo, hi = _ci(ab)
            r = dict(cell=cl, region=info[cl]["region"], c=info[cl]["c"], tau=info[cl]["tau"], measure=k,
                     alpha=al, alpha_lo=lo, alpha_hi=hi, value_at_pmax=float(sub.loc[ps[-1], k]))
            if k == "U":   # M5: extrapolated p at which U would fall below u_target (only if U is falling)
                r["p_U_below_target"] = float(np.exp((a - np.log(DESIGN["u_target"])) / al)) if al > 0 else np.inf
            rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "S2_ALPHAS.csv", index=False)

    win = [c for c in cell_list if info[c]["region"] == "window"]
    ctl = [c for c in cell_list if info[c]["region"] != "window"]
    aE = A[A.measure == "E"].set_index("cell")
    checks = []
    held1 = all(aE.loc[c, "alpha_lo"] > 0 for c in ctl)
    checks.append(("M1", "controls: E falls with p (alpha interval above 0) in both control cells",
                   "; ".join(f"{c}: {aE.loc[c, 'alpha']:.2f} [{aE.loc[c, 'alpha_lo']:.2f}, {aE.loc[c, 'alpha_hi']:.2f}]"
                             for c in ctl), held1))
    n_fall = int(sum(aE.loc[c, "alpha_lo"] > 0 for c in win))
    checks.append(("M2", f"window: E falls with p in >= {DESIGN['m2_min_cells']} of {len(win)} cells",
                   f"{n_fall} of {len(win)}", n_fall >= DESIGN["m2_min_cells"]))
    diff_b = np.median([alpha_b[(c, "E")] for c in win], 0) - np.median([alpha_b[(c, "E")] for c in ctl], 0)
    diff = float(np.median(aE.loc[win, "alpha"]) - np.median(aE.loc[ctl, "alpha"]))
    lo, hi = _ci(diff_b)
    checks.append(("M3", "slower in the window: median alpha(E) window - controls, interval below 0",
                   f"{diff:.3f} [{lo:.3f}, {hi:.3f}]", hi < 0))
    aU = A[A.measure == "U"].set_index("cell")
    med_b = np.median([alpha_b[(c, "U")] for c in win], 0)
    med = float(np.median(aU.loc[win, "alpha"]))
    lo, hi = _ci(med_b)
    checks.append(("M4", "labels: median alpha(U) over window cells > 0, interval above 0",
                   f"{med:.3f} [{lo:.3f}, {hi:.3f}]", lo > 0))
    K = pd.DataFrame(checks, columns=["check", "what", "value", "held"])
    K.to_csv(OUT / "S2_CHECKS.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"\nexponents alpha (larger = mean-field catches up faster), fitted on p = {ps}\n")
    tab = A.pivot_table(index=["region", "cell"], columns="measure", values="alpha")[MEASURES]
    print(tab.round(2).to_string())
    print("\n95% intervals for alpha(E) and alpha(U):")
    print(A[A.measure.isin(["E", "U"])][["cell", "region", "measure", "alpha", "alpha_lo", "alpha_hi"]]
          .round(3).to_string(index=False))
    print("\nM5 (reported): U at the largest p, and the p at which U would fall below "
          f"{DESIGN['u_target']} if the fitted power law held (an extrapolation):")
    print(aU[["region", "value_at_pmax", "p_U_below_target"]].round(3).to_string())
    print("\n=== checks ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S2_ALPHAS.csv'} and {OUT / 'S2_CHECKS.csv'}")


# ---------------------------------------------------------------------------
# amendments (post hoc, PLAN.md Amendment 1): A1 best mean-field setting, A2 800 rows
# ---------------------------------------------------------------------------

def amend_job(args):
    """One (p, cell) on AMEND_BLOCKS * 400 rows:
      U_same : the primary U (mean-field at the cell's own c, tau) recomputed on 800 rows (A2)
      U_best : for each exact teacher k, the best of the 60 mean-field menu settings,
               i.e. the setting with the smallest out-of-fold error for that teacher (A1).
               Picking the best on the same out-of-fold errors flatters mean-field slightly,
               which works AGAINST the claim that mean-field can't catch up.
    Per-sample squared residuals are saved so the exponents can be bootstrapped."""
    p, cl = args
    cell = cell_name(cl["c"], cl["tau"])
    f = OUT / "persample_amend" / f"p{p}_{cell}.npz"
    if not f.exists():
        nb = AMEND_BLOCKS
        BQ = bs.load_rows(BANKS, p, "Q", cl["c"], cl["tau"], nb)
        S = exact_scores(BQ, p, cell)
        d2 = (S - S.mean(0)) ** 2
        r2_same = oof_residuals(S, mf_design(bs.load_rows(BANKS, p, "MF", cl["c"], cl["tau"], nb), p))
        best_u = np.full(S.shape[1], np.inf)
        r2_best = np.empty_like(S)
        best_set = np.empty(S.shape[1], dtype=object)
        for c, t in MENU:
            r2 = oof_residuals(S, mf_design(bs.load_rows(BANKS, p, "MF", c, t, nb), p))
            u = r2.sum(0) / d2.sum(0)
            better = u < best_u
            best_u[better] = u[better]
            r2_best[:, better] = r2[:, better]
            best_set[better] = cell_name(c, t)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp.npz")
        np.savez_compressed(tmp, d2=d2, r2_same=r2_same, r2_best=r2_best, best_set=best_set.astype(str))
        tmp.replace(f)
    a = dict(np.load(f))
    u = lambda r2: float(np.median(r2.sum(0) / a["d2"].sum(0)))
    sets, counts = np.unique(a["best_set"], return_counts=True)
    return dict(p=p, cell=cell, region=cl["region"], U_same800=u(a["r2_same"]), U_best800=u(a["r2_best"]),
                best_setting=sets[np.argmax(counts)], n_rows=len(a["d2"]))


def amend_available_ps():
    ps = []
    for d in sorted(BANKS.glob("p*")):
        p = int(d.name[1:])
        need = [(k, cl["c"], cl["tau"]) for cl in cells() for k in ("Q", "MF")] + [("MF", c, t) for c, t in MENU]
        if all(bs.block_path(BANKS, p, k, c, t, b).exists() for k, c, t in need for b in range(AMEND_BLOCKS)):
            ps.append(p)
    return sorted(ps)


def do_amend(workers):
    ps = amend_available_ps()
    if not ps:
        raise SystemExit(f"STOP: no p has all amendment banks (run `banks --blocks {AMEND_BLOCKS} --menu`)")
    jobs = [(p, cl) for p in ps for cl in cells()]
    print(f"amendments: {len(jobs)} (p, cell) pairs for p = {ps}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        rows = [fu.result() for fu in as_completed([ex.submit(amend_job, j) for j in jobs])]
    M = pd.DataFrame(rows).sort_values(["region", "cell", "p"])
    P = pd.read_csv(OUT / "S2_MEASURES.csv")[["p", "cell", "U"]].rename(columns={"U": "U_same400"})
    M = M.merge(P, on=["p", "cell"], how="left")
    M.to_csv(OUT / "S2_AMEND.csv", index=False)

    fit_ps = [p for p in ps if p >= DESIGN["fit_p_min"]]
    info = {cell_name(cl["c"], cl["tau"]): cl for cl in cells()}
    rows, ab = [], {}
    for key in ("U_same800", "U_best800"):
        r2k = "r2_same" if key == "U_same800" else "r2_best"
        boot = {c: np.empty((DESIGN["n_boot"], len(fit_ps))) for c in info}
        for j, p in enumerate(fit_ps):
            A = {c: dict(np.load(OUT / "persample_amend" / f"p{p}_{c}.npz")) for c in info}
            m = len(next(iter(A.values()))["d2"])
            idx = np.random.default_rng(_seed(f"s2_amend_boot_p{p}")).integers(0, m, (DESIGN["n_boot"], m))
            for c, a in A.items():
                boot[c][:, j] = np.median(a[r2k][idx].sum(1) / a["d2"][idx].sum(1), axis=1)
        for c in info:
            sub = M[M.cell == c].set_index("p")
            al, _ = fit_alpha(fit_ps, sub.loc[fit_ps, key])
            ab[(c, key)] = np.array([fit_alpha(fit_ps, boot[c][b])[0] for b in range(DESIGN["n_boot"])])
            lo, hi = _ci(ab[(c, key)])
            rows.append(dict(cell=c, region=info[c]["region"], measure=key, alpha=al, alpha_lo=lo, alpha_hi=hi,
                             value_at_pmax=float(sub.loc[fit_ps[-1], key])))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "S2_AMEND_ALPHAS.csv", index=False)
    win = [c for c in info if info[c]["region"] == "window"]
    checks = []
    for key, tag in (("U_same800", "A2 (800 rows)"), ("U_best800", "A1 (best mean-field setting, 800 rows)")):
        aa = A[A.measure == key].set_index("cell")
        med_b = np.median([ab[(c, key)] for c in win], 0)
        lo, hi = _ci(med_b)
        med = float(np.median(aa.loc[win, "alpha"]))
        checks.append(("M4", f"under {tag}: median alpha(U) over window cells > 0, interval above 0",
                       f"{med:.3f} [{lo:.3f}, {hi:.3f}]", lo > 0))
    K = pd.DataFrame(checks, columns=["check", "what", "value", "held"])
    K.to_csv(OUT / "S2_AMEND_CHECKS.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nmedian over cells, by region and p (U = share of the exact score mean-field can't explain):")
    print(M.groupby(["region", "p"])[["U_same400", "U_same800", "U_best800"]].median()
          .to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nmost common best mean-field setting per window cell, at the largest p:")
    w = M[(M.region == "window") & (M.p == ps[-1])][["cell", "best_setting", "U_same800", "U_best800"]]
    print(w.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nexponents alpha for U (fitted on p = {fit_ps}):")
    print(A.pivot_table(index=["region", "cell"], columns="measure", values="alpha").round(3).to_string())
    print("\n=== amendment checks (post hoc; the original M4 result stands as reported) ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S2_AMEND.csv'}, {OUT / 'S2_AMEND_ALPHAS.csv'} and {OUT / 'S2_AMEND_CHECKS.csv'}")


# ---------------------------------------------------------------------------

def _pool(fn, jobs, workers, label):
    print(f"{len(jobs)} {label}, {workers} workers", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                st, nm = fu.result()
                if st != "skip" or i % 100 == 0:
                    print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(jobs)}] FAILED {futs[fu][1:]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry (finished pieces are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["time", "banks", "measure", "score", "amend"])
    ap.add_argument("--blocks", type=int, default=1, help="for `banks`: blocks of 400 rows (amendments: 2)")
    ap.add_argument("--menu", action="store_true", help="for `banks`: also the 60 mean-field menu settings")
    ap.add_argument("--ps", default=",".join(map(str, DESIGN["ps"])))
    ap.add_argument("--p", type=int, default=16, help="for `time`")
    ap.add_argument("--workers", type=int, default=31)
    a = ap.parse_args()
    if a.workers > 31:
        raise SystemExit("STOP: at most 31 workers (32-core limit)")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "time":
        do_time(a.p, a.workers)
    elif a.phase == "banks":
        do_banks([int(x) for x in a.ps.split(",")], a.workers, a.blocks, a.menu)
    elif a.phase == "measure":
        do_measure(a.workers)
    elif a.phase == "amend":
        do_amend(a.workers)
    else:
        do_score()


if __name__ == "__main__":
    main()
