"""
Fold protocol, classical comparators, LR evaluation, frontiers.

Faithful port of the evaluation machinery in 13_/14_.  Two input pipelines
existed and BOTH are preserved exactly, selected by `pipeline=`:

  pipeline="synthetic"  (13_):
      AQFM bank computed ONCE on the RAW, UNSCALED X (no clip), shared by folds.
      Classical maps: StandardScaler fit on train fold, no clip.
  pipeline="real"       (14_):
      Per fold: StandardScaler fit on train, clip to [-3, 3]; the SAME scaled
      inputs feed both the AQFM evolution (fold-specific banks) and classical maps.

See docs/PROTOCOL_NOTES.md for why this difference matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .aqfm import (AQFM_FAMILIES, LOCAL_FAMILIES, PAIR_RICH_FAMILIES,
                   compute_bank, family_matrix)

SEED = 42
J_GRID = (0.0, 1.0)
TAU_GRID = (0.1, 0.3, 1.0, 2.0, 5.0, 10.0)
N_STEPS = 400
C = 1.0
MAX_ITER = 10_000
INPUT_CLIP = 3.0
CLASSICAL_KINDS = ("Raw", "PolyInteractions", "Poly2Full", "Fourier", "RFF")


# ---------------------------------------------------------------------------
# Folds + preprocessing
# ---------------------------------------------------------------------------

def fold_splits(X, y, n_splits=5, seed=SEED):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def preprocess_fold_inputs(X, tr, te, clip=INPUT_CLIP):
    """Train-only StandardScaler, then clip (14_ real pipeline)."""
    sx = StandardScaler()
    Xtr = np.clip(sx.fit_transform(X[tr]), -clip, clip)
    Xte = np.clip(sx.transform(X[te]), -clip, clip)
    return Xtr, Xte


def fold_scaled_full_matrix(X, tr, te, clip=INPUT_CLIP):
    Xtr, Xte = preprocess_fold_inputs(X, tr, te, clip)
    Z = np.empty_like(X, dtype=float)
    Z[tr], Z[te] = Xtr, Xte
    return Z


# ---------------------------------------------------------------------------
# Classical comparators (verbatim)
# ---------------------------------------------------------------------------

def natural_dimensions(p):
    q = p * (p - 1) // 2
    return sorted(set([p, 3 * p, q, 3 * q, 3 * p + 3 * q]))


def comparator_dimensions(p):
    dims = set(natural_dimensions(p))
    dims |= {p, p + p * (p - 1) // 2, p + p * (p + 1) // 2}
    return sorted(dims)


def deterministic_fourier(X, D):
    """Axis-cycling sin(kx_j), cos(kx_j), k = 1, 2, ...; first D columns."""
    n, p = X.shape
    cols, harmonic = [], 1
    while len(cols) < D:
        for j in range(p):
            if len(cols) < D:
                cols.append(np.sin(harmonic * X[:, j]))
            if len(cols) < D:
                cols.append(np.cos(harmonic * X[:, j]))
        harmonic += 1
    return np.column_stack(cols[:D])


def classical_map(Ztr, Zte, D, kind):
    """Map ALREADY-preprocessed inputs (Ztr/Zte) to a D-dim classical basis.
    Returns None when `kind` has no natural size D (as in the originals)."""
    p = Ztr.shape[1]
    if kind == "Raw":
        return (Ztr, Zte) if D == p else None
    if kind in ("PolyInteractions", "Poly2Full"):
        poly = PolynomialFeatures(degree=2, interaction_only=(kind == "PolyInteractions"),
                                  include_bias=False)
        A, B = poly.fit_transform(Ztr), poly.transform(Zte)
        return (A, B) if A.shape[1] == D else None
    if kind == "Fourier":
        return deterministic_fourier(Ztr, D), deterministic_fourier(Zte, D)
    if kind == "RFF":
        rff = RBFSampler(gamma=1.0 / p, n_components=D, random_state=42)
        return rff.fit_transform(Ztr), rff.transform(Zte)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def fit_eval_lr(Atr, Ate, ytr, yte):
    """Feature-space StandardScaler -> LR(C=1). Binary or multiclass (14_ form;
    identical to 13_ for binary since predict == (p>=0.5))."""
    sc = StandardScaler()
    Atr = sc.fit_transform(Atr)
    Ate = sc.transform(Ate)
    m = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs").fit(Atr, ytr)
    prob_tr, prob_te = m.predict_proba(Atr), m.predict_proba(Ate)
    pred_te = m.predict(Ate)
    if len(m.classes_) == 2:
        tr_auc = roc_auc_score(ytr, prob_tr[:, 1])
        te_auc = roc_auc_score(yte, prob_te[:, 1])
    else:
        tr_auc = roc_auc_score(ytr, prob_tr, labels=m.classes_, multi_class="ovr",
                               average="macro")
        te_auc = roc_auc_score(yte, prob_te, labels=m.classes_, multi_class="ovr",
                               average="macro")
    return {"train_auc": tr_auc, "test_auc": te_auc,
            "test_accuracy": accuracy_score(yte, pred_te),
            "test_kappa": cohen_kappa_score(yte, pred_te)}


# ---------------------------------------------------------------------------
# AQFM banks for a dataset under either pipeline
# ---------------------------------------------------------------------------

def aqfm_banks(X, y, pipeline, Js=J_GRID, taus=TAU_GRID, n_steps=N_STEPS,
               cache=None):
    """Return {(fold, J, tau): bank} where bank rows align with X.

    synthetic: one bank on raw X, reused for every fold (fold key = 0).
    real:      one bank per fold on fold-scaled+clipped X.
    `cache` (dict) lets callers reuse banks across calls.
    """
    cache = {} if cache is None else cache
    splits = fold_splits(X, y)
    for J in Js:
        for tau in taus:
            if pipeline == "synthetic":
                key = (0, J, tau)
                if key not in cache:
                    cache[key] = compute_bank(X, J, tau, n_steps)
            else:
                for f, (tr, te) in enumerate(splits, 1):
                    key = (f, J, tau)
                    if key not in cache:
                        Z = fold_scaled_full_matrix(X, tr, te)
                        cache[key] = compute_bank(Z, J, tau, n_steps)
    return cache


# ---------------------------------------------------------------------------
# Full evaluation (same rows as the originals' fold_metrics.csv)
# ---------------------------------------------------------------------------

def evaluate(name, X, y, pipeline, banks, Js=J_GRID, taus=TAU_GRID,
             families=AQFM_FAMILIES, classical=CLASSICAL_KINDS):
    X, y = np.asarray(X, float), np.asarray(y, int)
    n, p = X.shape
    splits = fold_splits(X, y)
    rows = []
    for f, (tr, te) in enumerate(splits, 1):
        for J in Js:
            for tau in taus:
                bank = banks[(0 if pipeline == "synthetic" else f, J, tau)]
                for fam in families:
                    F = family_matrix(bank, fam, p)
                    if F.shape[1] == 0:
                        continue
                    rows.append({"dataset": name, "representation_type": "AQFM",
                                 "representation": fam, "D": F.shape[1], "J": J,
                                 "tau": tau, "fold": f,
                                 **fit_eval_lr(F[tr], F[te], y[tr], y[te])})
        if pipeline == "synthetic":
            sx = StandardScaler()
            Ztr, Zte = sx.fit_transform(X[tr]), sx.transform(X[te])
        else:
            Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        for D in comparator_dimensions(p):
            for kind in classical:
                z = classical_map(Ztr, Zte, D, kind)
                if z is None:
                    continue
                rows.append({"dataset": name, "representation_type": "classical",
                             "representation": kind, "D": D, "J": np.nan,
                             "tau": np.nan, "fold": f,
                             **fit_eval_lr(z[0], z[1], y[tr], y[te])})
    return pd.DataFrame(rows)


def summarize(fold_df):
    g = ["dataset", "representation_type", "representation", "D", "J", "tau"]
    return fold_df.groupby(g, dropna=False).agg(
        test_auc_mean=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
        train_auc_mean=("train_auc", "mean"),
        test_kappa_mean=("test_kappa", "mean"), test_kappa_sd=("test_kappa", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
    ).reset_index()


def build_frontiers(summary, metric="test_auc_mean"):
    """Descriptive post-hoc frontier (best observed mean per D per group).
    metric: 'test_auc_mean' (13_ synthetic) or 'test_kappa_mean' (14_ real)."""
    rows = []
    for ds in summary.dataset.unique():
        d = summary[summary.dataset == ds]
        aq, cl = d[d.representation_type == "AQFM"], d[d.representation_type == "classical"]
        groups = {
            "AQFM_local": aq[aq.representation.isin(LOCAL_FAMILIES)],
            "AQFM_pair_rich": aq[aq.representation.isin(PAIR_RICH_FAMILIES)],
            "AQFM_all": aq,
            "classical_generic": cl[cl.representation.isin(["Fourier", "RFF"])],
            "classical_all": cl,
        }
        for D in sorted(d.D.unique()):
            for label, q in groups.items():
                q = q[q.D == D]
                if q.empty:
                    continue
                r = q.loc[q[metric].idxmax()]
                rows.append({"dataset": ds, "D": int(D), "frontier_group": label,
                             "best": r[metric], "representation": r.representation,
                             "J": r.J, "tau": r.tau})
    return pd.DataFrame(rows)


# ===========================================================================
# PROTOCOL v2  (adopted 2026-10-07: decisions 1-3 in docs/PROTOCOL_NOTES.md)
#   1. integrator: unitary Strang splitting, n_steps = max(400, 160*tau);
#      J = 0 via exact product-state factorisation
#   2. ONE input pipeline for synthetic and real: per-fold train-only
#      StandardScaler + clip[-3,3], shared by AQFM and every classical map
#   3. classical comparators: original five PLUS FourierBalanced,
#      FourierLabelMatched, and (discrete-feature datasets only) OneHot,
#      LevelIndicator.  Originals retained for continuity.
# Everything else (Hamiltonian, J/tau grids, families, CV, LR) unchanged.
# ===========================================================================

from . import dictionaries as dic  # noqa: E402

V2_METHOD = "strang"
NEW_CLASSICAL = ("FourierBalanced", "FourierLabelMatched", "OneHot", "LevelIndicator")


def aqfm_banks_v2(X, y, Js=J_GRID, taus=TAU_GRID, cache=None):
    """Per-fold banks on fold-scaled+clipped inputs, Strang integrator."""
    cache = {} if cache is None else cache
    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        Z = fold_scaled_full_matrix(X, tr, te)
        for J in Js:
            for tau in taus:
                if (f, J, tau) not in cache:
                    cache[(f, J, tau)] = compute_bank(Z, J, tau, method=V2_METHOD)
    return cache


def evaluate_v2(name, X, y, banks, Js=J_GRID, taus=TAU_GRID,
                families=AQFM_FAMILIES):
    X, y = np.asarray(X, float), np.asarray(y, int)
    n, p = X.shape
    rows = []

    def add(rtype, rep, D, J, tau, f, met):
        rows.append({"dataset": name, "representation_type": rtype,
                     "representation": rep, "D": D, "J": J, "tau": tau,
                     "fold": f, **met})

    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        # ---- AQFM -------------------------------------------------------
        for J in Js:
            for tau in taus:
                bank = banks[(f, J, tau)]
                for fam in families:
                    F = family_matrix(bank, fam, p)
                    if F.shape[1]:
                        add("AQFM", fam, F.shape[1], J, tau, f,
                            fit_eval_lr(F[tr], F[te], y[tr], y[te]))
        # ---- classical (same fold-scaled + clipped inputs) ----------------
        Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        lmf = dic.LabelMatchedFourier().fit(Ztr, y[tr])
        dims = comparator_dimensions(p)
        for D in dims:
            for kind in CLASSICAL_KINDS:
                z = classical_map(Ztr, Zte, D, kind)
                if z is not None:
                    add("classical", kind, D, np.nan, np.nan, f,
                        fit_eval_lr(z[0], z[1], y[tr], y[te]))
            A, B = dic.balanced_fourier(Ztr, D), dic.balanced_fourier(Zte, D)
            add("classical", "FourierBalanced", D, np.nan, np.nan, f,
                fit_eval_lr(A, B, y[tr], y[te]))
            A, B = lmf.transform(Ztr, D), lmf.transform(Zte, D)
            if A is not None:
                add("classical", "FourierLabelMatched", D, np.nan, np.nan, f,
                    fit_eval_lr(A, B, y[tr], y[te]))
        if dic.has_discrete(Ztr):
            A, B = dic.one_hot(Ztr, Zte)
            add("classical", "OneHot", A.shape[1], np.nan, np.nan, f,
                fit_eval_lr(A, B, y[tr], y[te]))
            A, B = dic.level_indicator(Ztr, Zte, y[tr])
            add("classical", "LevelIndicator", A.shape[1], np.nan, np.nan, f,
                fit_eval_lr(A, B, y[tr], y[te]))
    return pd.concat([pd.DataFrame(rows), anova_rows_v2(name, X, y)],
                     ignore_index=True)


ORIGINAL_CLASSICAL = set(CLASSICAL_KINDS)


def build_frontiers_v2(summary, metric="test_auc_mean"):
    """Adds 'classical_original' and 'classical_fair' (all, incl. new) groups."""
    rows = []
    for ds in summary.dataset.unique():
        d = summary[summary.dataset == ds]
        aq, cl = d[d.representation_type == "AQFM"], d[d.representation_type == "classical"]
        groups = {
            "AQFM_local": aq[aq.representation.isin(LOCAL_FAMILIES)],
            "AQFM_pair_rich": aq[aq.representation.isin(PAIR_RICH_FAMILIES)],
            "AQFM_all": aq,
            "classical_original": cl[cl.representation.isin(ORIGINAL_CLASSICAL)],
            "classical_fair": cl,
        }
        for D in sorted(d.D.unique()):
            for label, q in groups.items():
                q = q[q.D == D]
                if q.empty:
                    continue
                r = q.loc[q[metric].idxmax()]
                rows.append({"dataset": ds, "D": int(D), "frontier_group": label,
                             "best": r[metric], "representation": r.representation,
                             "J": r.J, "tau": r.tau})
    return pd.DataFrame(rows)


def anova_rows_v2(name, X, y):
    """Fold rows for the ANOVA-matched control (appendable to v2 fold_metrics)."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    rows = []
    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        for k in dic.ANOVA_KINDS:
            A, B = dic.anova_matched(Ztr, k), dic.anova_matched(Zte, k)
            if A.shape[1] == 0:
                continue
            rows.append({"dataset": name, "representation_type": "classical",
                         "representation": f"Anova{k}", "D": A.shape[1],
                         "J": np.nan, "tau": np.nan, "fold": f,
                         **fit_eval_lr(A, B, y[tr], y[te])})
    return pd.DataFrame(rows)
