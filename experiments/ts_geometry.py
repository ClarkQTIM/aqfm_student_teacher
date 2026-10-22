#!/usr/bin/env python3
"""
Geometric difference between the map's kernel and classical / twin kernels
(Huang et al., "Power of data in quantum machine learning", Nat. Commun. 12,
2631 (2021), Eq. 5), on the tau sweep and on the real datasets.

Plan, definitions and predictions: docs/geometry_plan.md (written before any run).

No new quantum simulation: everything is built from banks that already exist.

Phases
  tt     tau sweep.  One job per (p, kappa, tau).  For every teacher's own
         kernel ("target"), g against classical RBF kernels on the inputs and
         against the twin maps (mean-field, J=0, exact), at 4 values of lambda.
         Label-aware model complexity s per teacher dataset.
           -> OUT/geometry/tt/<cell>.csv and <cell>_labels.csv   (restartable)
  real   real datasets (--group ophtho | pool).  One job per dataset.  The map
         kernel at all 60 (kappa, tau) settings with the full readout, plus at
         the exact student's own discovery choice.
           -> <realdata out>/<name>/geometry.csv (+ geometry_labels.csv)
  score  merges with the existing result tables and prints the checks G1-G9.
           -> OUT/geometry/GEO_TT.csv, <realdata out>/GEO_REAL.csv

Usage (on the server; see the run block in the chat for the full commands):
  python experiments/ts_geometry.py tt    --out runs/tau_sweep --workers 31
  python experiments/ts_geometry.py real  --group ophtho --workers 31
  python experiments/ts_geometry.py real  --group pool   --workers 31
  python experiments/ts_geometry.py score --out runs/tau_sweep
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: workers x 1 <= 32 cores

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ts_tau_sweep as tt                                        # noqa: E402
from aqfm_prescreen.aqfm import family_matrix                    # noqa: E402

GEO = {
    "version": "geometry_1",
    "plan": "docs/geometry_plan.md",
    # lambda is in units of the MEAN eigenvalue: every kernel is scaled to trace N, so its
    # eigenvalues average 1.  lambda = 0.01 means "ignore directions carrying < ~1% of an
    # average direction's weight".
    "lambdas": [1e-3, 1e-2, 1e-1, 1.0],
    "lam_primary": 1e-2,
    # RBF width: gamma = c / median squared distance between rows (the median heuristic),
    # three widths; the emulator is credited with whichever width is best (smallest g).
    "rbf_c": [0.25, 1.0, 4.0],
    "n_max_real": 600,                   # real data: subsample to the tau sweep's N (g grows with N)
    "eig_tol": 1e-10,                    # eigenvalues below tol * largest are treated as 0
    "gap": 0.05,                         # "short by >= 0.05 AUC", as everywhere else
    "auroc_min": 0.75,                   # G1, G2
    "spearman_min": 0.5,                 # G1, G2
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def standardize(F):
    """columns to mean 0, sd 1; constant columns dropped (they carry nothing).
    Same convention as the students, which standardize before the logistic regression."""
    F = np.asarray(F, float)
    sd = F.std(0)
    keep = sd > 1e-9
    if not keep.any():
        return None
    return (F[:, keep] - F[:, keep].mean(0)) / sd[keep]


def trace_normalize(K):
    """Huang et al. assume Tr(K) = N for every kernel (the scale of a kernel is arbitrary,
    and g is only meaningful once both kernels are on the same scale)."""
    return K * (len(K) / np.trace(K))


def linear_kernel(F):
    """K = F F^T on standardized features: exactly what a linear student on these features
    can express.  Rank <= number of features (at most 108 here), so it is low-rank."""
    Fs = standardize(F)
    if Fs is None:
        return None
    return trace_normalize(Fs @ Fs.T)


def rbf_kernels(F, cs=None):
    """Gaussian kernels exp(-gamma |f - f'|^2) on standardized features, one per width c,
    gamma = c / median squared distance.  Diagonal = 1, so trace = N already.  Full rank:
    a Gaussian kernel can fit any smooth function of its inputs, given enough data."""
    cs = cs or GEO["rbf_c"]
    Fs = standardize(F)
    if Fs is None:
        return None
    sq = (Fs ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Fs @ Fs.T, 0.0)
    med = np.median(d2[np.triu_indices(len(d2), 1)])
    med = med if med > 0 else 1.0
    return {c: np.exp(-(c / med) * d2) for c in cs}


class Eig:
    """eigen-decomposition of a PSD kernel, K = V diag(e) V^T, tiny/negative e set to 0"""

    def __init__(self, K):
        e, V = eigh(K)
        e = np.where(e > GEO["eig_tol"] * max(e.max(), 1e-300), e, 0.0)
        self.e, self.V, self.N = e, V, len(K)

    def sqrt_factor(self):
        """B with K = B B^T and B = V_r diag(sqrt(e_r)): only the nonzero directions,
        so a rank-24 kernel gives an N x 24 factor (cheap)."""
        r = self.e > 0
        return self.V[:, r] * np.sqrt(self.e[r])


# ---------------------------------------------------------------------------
# the two quantities
# ---------------------------------------------------------------------------

def geometric_difference(target: Eig, emul: Eig, lam):
    """Regularized geometric difference of the emulator kernel K_C from the target K_T, and
    the matching training-error coefficient (derivation: docs/geometry_plan.md, Section 2):

        g_lam^2 = || sqrt(K_T) sqrt(K_C) (K_C + lam)^-2 sqrt(K_C) sqrt(K_T) ||
        t_lam   = || sqrt(K_T) lam^2 (K_C + lam)^-2 sqrt(K_T) || / N

    For ANY labels y = sqrt(K_T) v the target can express (complexity s_T = |v|^2), kernel
    ridge regression with the emulator and ridge lam has
        complexity          s_C <= g_lam^2 * s_T      (cost in data:  ~g^2 times more samples)
        mean training error     <= t_lam  * s_T       (what it fails to fit at all)
    As lam -> 0 with K_C invertible, g_lam -> Huang's Eq. 5 and t_lam -> 0.

    Implementation (avoids N x N products).  Write sqrt(K_T) = U diag(sqrt e_T) U^T = BT U^T,
    with U the target's nonzero eigenvectors (N x r) and BT = U diag(sqrt e_T).  Any matrix
    M = V_C diag(m) V_C^T that is diagonal in the emulator's eigenbasis then gives
        sqrt(K_T) M sqrt(K_T) = U (A^T diag(m) A) U^T,   A = V_C^T BT  (N x r),
    whose norm is || A^T diag(m) A || = || diag(sqrt m) A ||^2 (U has orthonormal columns).
    So we need one N x r matrix A and two spectral norms of row-scaled copies of it."""
    A = emul.V.T @ target.sqrt_factor()               # N x r: the target's directions, emulator's basis
    m_g = emul.e / (emul.e + lam) ** 2                # what the emulator can fit, per direction, at a cost
    m_t = lam ** 2 / (emul.e + lam) ** 2              # what it leaves unfitted, per direction
    g2 = np.linalg.norm(np.sqrt(m_g)[:, None] * A, 2) ** 2
    t = np.linalg.norm(np.sqrt(m_t)[:, None] * A, 2) ** 2 / target.N
    return float(np.sqrt(g2)), float(t)


def geometric_difference_eq5(KT, KC):
    """Huang et al. Eq. 5, unregularized: g = sqrt(|| sqrt(K_T) K_C^-1 sqrt(K_T) ||).
    Only used by the tests (needs an invertible K_C)."""
    eT = Eig(KT)
    S = (eT.V * np.sqrt(eT.e)) @ eT.V.T
    return float(np.sqrt(np.linalg.norm(S @ np.linalg.solve(KC, S), 2)))


def complexity(emul: Eig, y, lam):
    """label-aware: kernel ridge with ridge lam on labels y (+-1).
        s     = |w|^2 = y^T (K + lam)^-1 K (K + lam)^-1 y     (model complexity, Huang's s_K)
        train = lam^2 |(K + lam)^-1 y|^2 / N                   (mean squared training error)"""
    c = emul.V.T @ y
    s = float(np.sum(c ** 2 * emul.e / (emul.e + lam) ** 2))
    tr = float(np.sum(c ** 2 * lam ** 2 / (emul.e + lam) ** 2) / emul.N)
    return s, tr


def compare(target_K, emulators, extra=None):
    """g and t for one target kernel against a dict {(emulator, c): K} at every lambda"""
    eT = Eig(target_K)
    rows = []
    for (em, c), K in emulators.items():
        eC = Eig(K)
        for lam in GEO["lambdas"]:
            g, t = geometric_difference(eT, eC, lam)
            rows.append(dict(**(extra or {}), emulator=em, c=c, lam=lam, g=g, t=t,
                             rank_T=int((eT.e > 0).sum())))
    return rows


def label_rows(kernels, y01, extra):
    """complexity s and training error of each kernel {(kernel, c): K} for labels y (0/1)"""
    y = 2.0 * np.asarray(y01, float) - 1.0
    rows = []
    for (kn, c), K in kernels.items():
        e = Eig(K)
        for lam in GEO["lambdas"]:
            s, tr = complexity(e, y, lam)
            rows.append(dict(**extra, kernel=kn, c=c, lam=lam, s=s, train=tr))
    return rows


def _rbf_dict(name, F):
    Ks = rbf_kernels(F)
    return {} if Ks is None else {(name, c): K for c, K in Ks.items()}


# ---------------------------------------------------------------------------
# tau sweep
# ---------------------------------------------------------------------------

_CLS = {}          # per process: classical RBF kernels on the inputs, one set per p


def _classical(p):
    if p not in _CLS:
        _, Z = tt.inputs(p)                 # the SAME 600 x p inputs the tau sweep used
        _CLS[p] = _rbf_dict("classical", Z)
    return _CLS[p]


def tt_cell_job(args):
    """One (p, kappa, tau) cell of the tau sweep.
    kappa > 0: targets = exact (Q) and mean-field (MF) teachers, both readouts.
               emulators = classical RBF on z; RBF on each twin's features (the other
               coupled kind and J=0, same readout); linear kernel on the twin features.
    kappa = 0: targets = J=0 teachers; emulators = classical only."""
    out, p, kappa, tau, teachers = args
    gdir = out / "geometry" / "tt"
    cell = f"p{p}_k{kappa:g}_t{tau:g}"
    f = gdir / f"{cell}.csv"
    if f.exists():
        return cell, "skip"
    t0 = time.time()
    kinds = ["J0"] if kappa == 0 else ["Q", "MF"]
    banks = {k: tt.load_bank(out, p, k, kappa, tau) for k in kinds}
    if kappa > 0:
        banks["J0"] = tt.load_bank(out, p, "J0", 0.0, tau)
    rows, lrows = [], []
    for kind in kinds:
        for ro in ("local", "zlocal"):
            F_T = tt.readout_matrix(banks[kind], ro, p)
            K_T = linear_kernel(F_T)
            if K_T is None:                  # e.g. a readout that is constant: no labels were made
                continue
            em = dict(_classical(p))
            twins = [] if kind == "J0" else [k for k in ("Q", "MF", "J0") if k != kind]
            for tw in twins:
                F_tw = tt.readout_matrix(banks[tw], ro, p)
                em.update(_rbf_dict(f"{tw}_rbf", F_tw))
                K_lin = linear_kernel(F_tw)
                if K_lin is not None:
                    em[(f"{tw}_lin", 0.0)] = K_lin
            key = dict(kind=kind, p=p, kappa=kappa, tau=tau, readout=ro)
            rows += compare(K_T, em, key)
            # label-aware: every teacher dataset made from this target (seeds share the kernel)
            names = teachers[(teachers.kind == kind) & (teachers.readout == ro)].name
            for nm in names:
                lf = out / "labels" / f"{nm}.npy"
                if not lf.exists():
                    continue
                kern = {("target", 0.0): K_T, **em}
                lrows += label_rows(kern, np.load(lf), dict(name=nm, **key))
    gdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lrows).to_csv(gdir / f"{cell}_labels.csv", index=False)
    tmp = f.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(f)                                    # the targets file last: it marks the cell done
    return cell, f"ok {time.time() - t0:.0f}s"


def do_tt(out: Path, workers: int):
    M = pd.read_csv(out / "TT_TEACHERS.csv")
    M = M[~M.degenerate.astype(bool)]
    jobs = []
    for (p, kap, tau), g in M.groupby(["p", "kappa", "tau"]):
        jobs.append((out, int(p), float(kap), float(tau), g[["name", "kind", "readout"]].copy()))
    jobs.sort(key=lambda j: -j[1])                    # big p first
    _pool(tt_cell_job, jobs, workers, "tau-sweep cells")
    (out / "geometry" / "GEO_PROTOCOL.json").write_text(json.dumps(
        {"design": GEO, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

def _real_rows(n, name, nmax):
    """all rows if n <= nmax, else nmax rows drawn without replacement, seeded by the name"""
    if n <= nmax:
        return np.arange(n)
    rng = np.random.default_rng(tt._seed(f"geometry_rows_{name}"))
    return np.sort(rng.choice(n, nmax, replace=False))


def _sq_choice(o):
    """the exact student's most frequent discovery choice, e.g. 'XYZ|k2|tau5'"""
    R = pd.read_csv(o / "discovery_students.csv")
    ch = R[R.arm == "S_Q"].choice.mode().iloc[0]
    fam, k, t = ch.split("|")
    return fam, float(k[1:]), float(t[3:])


def real_job(args):
    """One real dataset.  Targets: the exact map's linear kernel (full readout) at each of the
    60 (kappa, tau) settings, and at the exact student's own discovery choice.  Emulators:
    classical RBF on the inputs, RBF + linear on mean-field and J=0 features (same readout)."""
    out, name = args
    o = out / name
    f = o / "geometry.csv"
    if f.exists():
        return name, "skip"
    t0 = time.time()
    P = np.load(o / "prepared.npz")
    Z, y = P["Z"], P["y"]
    idx = _real_rows(len(y), name, GEO["n_max_real"])
    Z, y = Z[idx], y[idx]
    p = Z.shape[1]
    D = tt.DESIGN
    bank = lambda kind, kap, tau: np.load(o / "banks" / f"{tt.bank_key(kind, kap, tau)}.npy")[idx]
    cls = _rbf_dict("classical", Z)
    rows, lrows = [], []

    def one(fam, kap, tau, setting):
        F_T = family_matrix(bank("Q", kap, tau), fam, p)
        K_T = linear_kernel(F_T)
        if K_T is None:
            return None
        em = dict(cls)
        for tw, (tk, tt_) in (("MF", (kap, tau)), ("J0", (0.0, tau))):
            F_tw = family_matrix(bank(tw, tk, tt_), fam, p)
            em.update(_rbf_dict(f"{tw}_rbf", F_tw))
            K_lin = linear_kernel(F_tw)
            if K_lin is not None:
                em[(f"{tw}_lin", 0.0)] = K_lin
        rows.extend(compare(K_T, em, dict(name=name, setting=setting, family=fam, kappa=kap, tau=tau,
                                          n_used=len(y))))
        return K_T, em

    for kap in D["student_kappas"]:
        for tau in D["student_taus"]:
            one("full", kap, tau, "grid")
    fam, kap, tau = _sq_choice(o)
    got = one(fam, kap, tau, "chosen")
    if got is not None:                                # label-aware, descriptive only
        K_T, em = got
        lrows += label_rows({("target", 0.0): K_T, **em}, y,
                            dict(name=name, setting="chosen", family=fam, kappa=kap, tau=tau))
    pd.DataFrame(lrows).to_csv(o / "geometry_labels.csv", index=False)
    tmp = f.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(f)
    return name, f"ok {time.time() - t0:.0f}s"


def _real_out(group):
    return Path("runs/realdata_pool" if group == "pool" else "runs/realdata")


def _real_names(out):
    names = []
    for d in sorted(out.iterdir()) if out.exists() else []:
        pf = d / "PREP.json"
        if pf.exists() and json.loads(pf.read_text())["status"] == "ok" \
                and (d / "discovery_students.csv").exists():
            names.append(d.name)
    return names


def do_real(out: Path, workers: int):
    names = _real_names(out)
    _pool(real_job, [(out, n) for n in names], workers, f"real datasets in {out}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _best_over_c(G, by, lam):
    """per target and emulator family: the emulator's best width (smallest g) at lambda"""
    g = G[G.lam == lam].copy()
    g["fam"] = g.emulator
    i = g.groupby(by + ["fam"]).g.idxmin()
    b = g.loc[i, by + ["fam", "g", "t", "c"]]
    w = b.pivot_table(index=by, columns="fam", values=["g", "t"])
    w.columns = [f"{a}_{f}" for a, f in w.columns]
    return w.reset_index()


def _auroc(score, label):
    ok = score.notna() & label.notna()
    s, l = score[ok], label[ok].astype(bool)
    if l.nunique() < 2:
        return np.nan, int(l.sum()), int(len(l))
    return float(roc_auc_score(l, s)), int(l.sum()), int(len(l))


def score_tt(out: Path, lam):
    files = sorted((out / "geometry" / "tt").glob("p*_k*_t*.csv"))
    files = [f for f in files if not f.name.endswith("_labels.csv")]
    if not files:
        raise SystemExit(f"STOP: no tau-sweep geometry files in {out / 'geometry' / 'tt'}")
    G = pd.concat([pd.read_csv(f) for f in files])
    key = ["kind", "p", "kappa", "tau", "readout"]
    W = _best_over_c(G, key, lam)
    af = out / "ADDONS_DATASETS.csv"
    if not af.exists():
        raise SystemExit(f"STOP: {af} not found (run ts_addons.py score first)")
    T = pd.read_csv(af)
    T["cls_gap"] = T.best_student_auc - T[["auc_classical_best", "auc_nonlinear_best"]].max(axis=1)
    T = T.merge(W, on=key, how="left")
    return G, T


def checks_tt(T, lam, tag=""):
    q = T[(T.kind == "Q") & (T.readout == "local")]
    rows = []
    a, k, n = _auroc(q.g_classical, q.cls_gap >= GEO["gap"])
    r = spearmanr(q.g_classical, q.cls_gap, nan_policy="omit").correlation
    rows.append(("G1", "classical: AUROC of g for 'classical short by >= 0.05' (exact teachers, local)",
                 a, f"{k}/{n} short", f">= {GEO['auroc_min']}", a >= GEO["auroc_min"]))
    rows.append(("G1", "classical: Spearman(g, classical shortfall)", r, "", f">= {GEO['spearman_min']}",
                 r >= GEO["spearman_min"]))
    a, k, n = _auroc(q.g_MF_rbf, q.gap_Q_MF >= GEO["gap"])
    r = spearmanr(q.g_MF_rbf, q.gap_Q_MF, nan_policy="omit").correlation
    rows.append(("G2", "twin: AUROC of g(MF||Q) for 'exact - mean-field >= 0.05'", a, f"{k}/{n} ahead",
                 f">= {GEO['auroc_min']}", a >= GEO["auroc_min"]))
    rows.append(("G2", "twin: Spearman(g(MF||Q), exact - mean-field)", r, "", f">= {GEO['spearman_min']}",
                 r >= GEO["spearman_min"]))
    win, below = q[q.region == "window"], q[q.region.str.startswith("below")]
    ratio = win.g_MF_rbf.median() / below.g_MF_rbf.median()
    rows.append(("G3", "twin: median g(MF||Q), window / weak-or-short region", ratio, "", ">= 2", ratio >= 2))
    # G4: Z-only readout control, same (p, kappa, tau) cells, window cells only
    qa = T[(T.kind == "Q") & (T.kappa >= 2) & (T.tau >= 3) & (T.tau <= 20)]
    cell = qa.groupby(["p", "kappa", "tau", "readout"]).g_classical.median().unstack("readout")
    if {"local", "zlocal"} <= set(cell.columns):
        fr = float((cell.zlocal < cell.local).mean())
    else:
        fr = np.nan
    rows.append(("G4", "Z-only control: share of window cells with g(Z-only) < g(full readout)", fr,
                 f"{len(cell)} cells", ">= 0.8", fr >= 0.8))
    mfw = T[(T.kind == "MF") & (T.readout == "local") & (T.region == "window")]
    ratio = mfw.g_classical.median() / win.g_classical.median()
    rows.append(("G5", "mean-field teachers are classically hard too: median g_classical, MF / exact (window)",
                 ratio, "", ">= 0.5", ratio >= 0.5))
    K = pd.DataFrame(rows, columns=["check", "what", "value", "count", "prediction", "held"])
    K.insert(0, "lambda", lam)
    return K


def score_real(group, lam, g0):
    out = _real_out(group)
    names = _real_names(out)
    Gs = [pd.read_csv(out / n / "geometry.csv") for n in names if (out / n / "geometry.csv").exists()]
    if not Gs:
        print(f"(no real-data geometry in {out})")
        return None
    G = pd.concat(Gs)
    key = ["name", "setting", "family", "kappa", "tau"]
    W = _best_over_c(G, key, lam)
    grid = W[W.setting == "grid"].groupby("name").agg(g_classical_max=("g_classical", "max"),
                                                      g_MF_rbf_max=("g_MF_rbf", "max"))
    ch = W[W.setting == "chosen"].set_index("name")[["g_classical", "g_MF_rbf", "t_MF_lin"]] \
        .add_suffix("_chosen")
    n_used = G.groupby("name").n_used.first()
    R = grid.join(ch).join(n_used)
    L = [pd.read_csv(out / n / "geometry_labels.csv") for n in names if (out / n / "geometry_labels.csv").exists()]
    if L:
        L = pd.concat(L)
        L = L[L.lam == lam]
        s = L.groupby(["name", "kernel"]).s.min().unstack()          # best width per kernel family
        R = R.join(pd.DataFrame({"s_target": s.get("target"), "s_classical": s.get("classical"),
                                 "s_MF_rbf": s.get("MF_rbf")}))
        R["s_ratio_MF"] = R.s_MF_rbf / R.s_target
        R["s_ratio_classical"] = R.s_classical / R.s_target
    T = pd.read_csv(out / "REAL_DATASETS.csv").set_index("name")
    R = R.join(T[[c for c in ["n", "p", "verdict", "margin_conf_lo", "confirmed_compact", "boolean_rule"] if c in T]])
    R["below_g0"] = R.g_classical_max < g0
    R.insert(0, "group", group)
    return R.reset_index()


def do_score(out: Path):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    lam0 = GEO["lam_primary"]
    G, T = score_tt(out, lam0)
    T.to_csv(out / "geometry" / "GEO_TT.csv", index=False)
    print(f"\n=== tau sweep: {len(T)} teacher datasets, lambda = {lam0} (primary) ===")
    q = T[T.kind == "Q"]
    print("\nexact teachers: median g (best width) by region and readout")
    print(q.groupby(["readout", "region"])[["g_classical", "g_MF_rbf", "g_J0_rbf", "t_MF_lin", "cls_gap",
                                            "gap_Q_MF"]].median().round(3).to_string())
    mf = T[T.kind == "MF"]
    print("\nmean-field teachers: median g by region (local readout)")
    print(mf[mf.readout == "local"].groupby("region")[["g_classical", "g_Q_rbf", "cls_gap"]]
          .median().round(3).to_string())
    print("\nG6 (descriptive): g^2 = predicted extra-data factor, window cells, exact teachers, local, by p")
    w = q[(q.readout == "local") & (q.region == "window")]
    print(w.groupby("p")[["g_classical", "g_MF_rbf"]].median().pow(2).round(1)
          .rename(columns=lambda c: c + "^2").to_string())
    K = checks_tt(T, lam0)
    print("\n=== checks (primary lambda) ===")
    print(K.to_string(index=False))
    print("\nsensitivity (other lambdas; not the test):")
    for lam in GEO["lambdas"]:
        if lam == lam0:
            continue
        _, Tl = score_tt(out, lam)
        print(checks_tt(Tl, lam)[["lambda", "check", "what", "value", "held"]].to_string(index=False, header=False))
    # threshold for the real-data rule-out (G7): 5th percentile of g_classical among tau-sweep
    # datasets (all teacher kinds and readouts) that the classical opponents miss by >= 0.05
    hard = T[T.cls_gap >= GEO["gap"]].g_classical.dropna()
    g0 = float(np.percentile(hard, 5)) if len(hard) else np.nan
    print(f"\nG7 threshold g0 = 5th percentile of g_classical over {len(hard)} classically-hard "
          f"tau-sweep datasets = {g0:.3f}")
    Rs = [r for r in (score_real(gr, lam0, g0) for gr in ("ophtho", "pool")) if r is not None]
    if Rs:
        R = pd.concat(Rs)
        R.to_csv(Path("runs") / "GEO_REAL.csv", index=False)
        cols = [c for c in ["group", "name", "n_used", "p", "verdict", "confirmed_compact", "g_classical_max",
                            "g_MF_rbf_max", "g_classical_chosen", "g_MF_rbf_chosen", "s_ratio_MF",
                            "s_ratio_classical", "below_g0"] if c in R]
        print("\n=== real datasets (lambda = primary) ===")
        print(R.sort_values("g_classical_max", ascending=False)[cols].round(3).to_string(index=False))
        b = R[R.below_g0]
        bad = b[(b.verdict != "classical enough") | (b.margin_conf_lo > 0)]
        print(f"\nG7: {len(b)} of {len(R)} real datasets below g0; among them, not 'classical enough' or a "
              f"confirmed lead: {len(bad)}  (prediction: 0)  -> {'held' if len(bad) == 0 else 'FAILED'}"
              + (f"  [{', '.join(bad.name)}]" if len(bad) else ""))
        pool = R[R.group == "pool"].dropna(subset=["s_ratio_MF"])
        if len(pool) and "pmlb_monk2" in set(pool.name):
            rk = int(pool.s_ratio_MF.rank(ascending=False)[pool.name == "pmlb_monk2"].iloc[0])
            print(f"G8: monk2's s(MF)/s(exact) ranks {rk} of {len(pool)} pool datasets "
                  f"(prediction: top 3) -> {'held' if rk <= 3 else 'FAILED'}")
        print(f"G9 (descriptive): share of real datasets below g0 = {R.below_g0.mean():.0%}")
        print("\nwrote runs/GEO_REAL.csv")
    print(f"wrote {out / 'geometry' / 'GEO_TT.csv'}")


# ---------------------------------------------------------------------------

def _pool(fn, jobs, workers, label):
    print(f"{len(jobs)} {label}, {workers} workers", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, st = fu.result()
                print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(jobs)}] FAILED\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry (finished jobs are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["tt", "real", "score"])
    ap.add_argument("--out", type=Path, default=Path("runs/tau_sweep"), help="tau-sweep output folder")
    ap.add_argument("--group", choices=["ophtho", "pool"], default="ophtho")
    ap.add_argument("--workers", type=int, default=31)
    a = ap.parse_args()
    if a.workers > 31:
        raise SystemExit("STOP: at most 31 workers (32-core limit)")
    if a.phase == "tt":
        do_tt(a.out, a.workers)
    elif a.phase == "real":
        do_real(_real_out(a.group), a.workers)
    else:
        do_score(a.out)


if __name__ == "__main__":
    main()
