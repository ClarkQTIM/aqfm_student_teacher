"""
Screen measures (docs/screen_plan.md, Amendments 1-3): cheap tests on (x, y) only.

RULE: this module sees the inputs Z and the labels y, nothing else.  It does not
import the map code (aqfm_prescreen), the teacher definitions, or any saved bank,
and no function here takes a teacher score or an oracle AUC.  tests/test_screen.py
checks this by inspecting the imports.

Every measure uses the SAME 5 outer folds as the students (StratifiedKFold,
shuffle, seed 42) and, where it needs them, the same 3 inner folds (seed 0).

  L1  learning-curve slope.  Three fixed classical learners are refit on 25, 50,
      75 and 100% of each outer-train fold (nested random subsets, 3 repeats) and
      scored on the outer-test fold.  L1 = AUC gain per doubling of the training
      set, from 50% to 100%, for the learner that is best at 100%.
      Large: still learning.  Near zero: flat (noise-limited, or already done).
      Also reported, exploratory: an asymptote estimate in the spirit of
      Cortes et al. (1994), from the train and test curves together.
  L2  stacking gain.  Six classical learners give out-of-fold scores on the
      inner folds; a logistic-regression stacker combines them.  L2 = outer-test
      AUC of the stack minus that of the single learner the inner folds rank best
      (Wolpert 1992).  Large: different learners catch different pieces.
  L3  nearest-neighbour bound (Cover & Hart 1967), baseline only.  From the
      leave-one-out 1-NN error R, the best possible error is at least
      (1 - sqrt(1 - 2R)) / 2 (asymptotically); reported as an ACCURACY ceiling.
  F1  interaction ladder.  Logistic regression on explicit expansions:
      additive splines -> + pairwise products -> + three-way products, then
      unrestricted (best of boosted trees, RBF SVM); outer-CV AUC of each.  The
      order is the first rung within 0.02 of the best.  Describes WHAT KIND of
      resistance, not part of the flag.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC

OUTER, INNER, CV_SEED, INNER_SEED = 5, 3, 42, 0          # identical to the student runs
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
L1_REPEATS = 3
ORDER_TOL = 0.02          # F1: a rung "already has it" if within this of the best rung
SIGNAL_MIN = 0.03         # F1: if no rung beats 0.5 by this much, order = "none found"


# ---------------------------------------------------------------------------
# learners (all fixed settings: these are measuring instruments, not students)
# ---------------------------------------------------------------------------

def _poly3(p):
    return make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=5000))


def _hgb(**kw):
    base = dict(learning_rate=0.1, max_leaf_nodes=15, max_iter=200, early_stopping=False, random_state=0)
    base.update(kw)
    return HistGradientBoostingClassifier(**base)


def _svm(p):
    return SVC(C=1.0, gamma=1.0 / p, kernel="rbf")


def _linear(p):
    return LogisticRegression(max_iter=2000)


def _splines(p):
    return make_pipeline(SplineTransformer(n_knots=9, degree=3), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=5000))


def _knn(p):
    return KNeighborsClassifier(25)


L1_LEARNERS = {"poly3": _poly3, "hgb": lambda p: _hgb(), "svm": _svm}
L2_LEARNERS = {"linear": _linear, "poly3": _poly3, "splines": _splines,
               "hgb": lambda p: _hgb(), "svm": _svm, "knn": _knn}


def _score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def _fit(make, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make().fit(X, y)


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def outer_folds(Z, y):
    return list(StratifiedKFold(OUTER, shuffle=True, random_state=CV_SEED).split(Z, y))


def _inner_folds(y):
    return list(StratifiedKFold(INNER, shuffle=True, random_state=INNER_SEED).split(np.zeros(len(y)), y))


def _rng(name, *k):
    import hashlib
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (name,) + k)).encode()).hexdigest()[:8], 16))


# ---------------------------------------------------------------------------
# L1: learning curves
# ---------------------------------------------------------------------------

def _asymptote(n, test, train):
    """Cortes et al. (1994)-style: test(n) = A - b n^-a, train(n) = A + c n^-g with a
    SHARED asymptote A; b, c >= 0.  Least squares on the averaged curves.
    Exploratory: with four sizes the fit is loose."""
    n = np.asarray(n, float) / n[-1]

    def res(q):
        A, b, a, c, g = q
        return np.r_[test - (A - b * n ** -a), train - (A + c * n ** -g)]
    q0 = [0.5 * (test[-1] + train[-1]), 0.05, 0.5, 0.05, 0.5]
    try:
        r = least_squares(res, q0, bounds=([0.5, 0, 0.1, 0, 0.1], [1.0, 1.0, 2.0, 1.0, 2.0]))
        return float(r.x[0])
    except Exception:
        return np.nan


def measure_L1(Z, y, name, folds):
    p = Z.shape[1]
    test = {k: np.zeros(len(FRACTIONS)) for k in L1_LEARNERS}
    train = {k: np.zeros(len(FRACTIONS)) for k in L1_LEARNERS}
    sizes = np.zeros(len(FRACTIONS))
    cnt = 0
    for fo, (tr, te) in enumerate(folds):
        for rep in range(L1_REPEATS):
            perm = _rng(name, "L1", fo, rep).permutation(tr)      # nested: each size is a prefix
            for i, f in enumerate(FRACTIONS):
                sub = perm[: int(round(f * len(tr)))]
                sizes[i] = len(sub)
                for k, mk in L1_LEARNERS.items():
                    m = _fit(lambda: mk(p), Z[sub], y[sub])
                    test[k][i] += _auc(y[te], _score(m, Z[te]))
                    train[k][i] += _auc(y[sub], _score(m, Z[sub]))
            cnt += 1
    for k in L1_LEARNERS:
        test[k] /= cnt
        train[k] /= cnt
    best = max(L1_LEARNERS, key=lambda k: test[k][-1])
    i50, i100 = FRACTIONS.index(0.5), FRACTIONS.index(1.0)
    out = {"L1_slope": test[best][i100] - test[best][i50],          # gain per doubling, 50% -> 100%
           "L1_learner": best,
           "L1_auc_full": test[best][i100],
           "L1_asymptote": _asymptote(sizes, test[best], train[best])}
    for k in L1_LEARNERS:
        for f, v in zip(FRACTIONS, test[k]):
            out[f"L1_curve_{k}_{int(f * 100)}"] = v
    return out


# ---------------------------------------------------------------------------
# L2: stacking gain
# ---------------------------------------------------------------------------

def _std_cols(A, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-12
    return (A - mu) / sd


def measure_L2(Z, y, name, folds):
    p = Z.shape[1]
    gains, stack_aucs, single_aucs, picks = [], [], [], []
    for tr, te in folds:
        Ztr, ytr = Z[tr], y[tr]
        oof = np.zeros((len(tr), len(L2_LEARNERS)))
        for a, b in _inner_folds(ytr):
            for j, mk in enumerate(L2_LEARNERS.values()):
                oof[b, j] = _score(_fit(lambda: mk(p), Ztr[a], ytr[a]), Ztr[b])
        inner_auc = [_auc(ytr, oof[:, j]) for j in range(oof.shape[1])]
        jbest = int(np.nanargmax(inner_auc))
        stacker = _fit(lambda: LogisticRegression(C=1.0, max_iter=2000), _std_cols(oof, oof), ytr)
        full = np.column_stack([_score(_fit(lambda: mk(p), Ztr, ytr), Z[te]) for mk in L2_LEARNERS.values()])
        a_stack = _auc(y[te], _score(stacker, _std_cols(full, oof)))
        a_single = _auc(y[te], full[:, jbest])
        gains.append(a_stack - a_single)
        stack_aucs.append(a_stack)
        single_aucs.append(a_single)
        picks.append(list(L2_LEARNERS)[jbest])
    return {"L2_gain": float(np.mean(gains)), "L2_stack_auc": float(np.mean(stack_aucs)),
            "L2_single_auc": float(np.mean(single_aucs)),
            "L2_single_pick": max(set(picks), key=picks.count)}


# ---------------------------------------------------------------------------
# L3: nearest-neighbour bound
# ---------------------------------------------------------------------------

def measure_L3(Z, y):
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    _, idx = nn.kneighbors(Z)                       # idx[:, 0] is the point itself
    R = float((y[idx[:, 1]] != y).mean())           # leave-one-out 1-NN error
    bayes_err_lower = (1 - np.sqrt(max(0.0, 1 - 2 * min(R, 0.5)))) / 2
    return {"L3_nn_error": R, "L3_ceiling_acc": 1 - bayes_err_lower}


# ---------------------------------------------------------------------------
# F1: interaction ladder
# ---------------------------------------------------------------------------

class _Expand:
    """explicit feature expansion for the interaction ladder: per-feature splines
    (any additive shape), optionally plus all products of 2 and/or 3 distinct
    (standardized) inputs.  Fit inside each training fold."""

    def __init__(self, orders):
        self.orders = orders

    def fit(self, Z, y=None):
        self.spl = SplineTransformer(n_knots=9, degree=3).fit(Z)
        return self

    def transform(self, Z):
        parts = [self.spl.transform(Z)]
        for k in self.orders:
            parts.append(np.column_stack([np.prod(Z[:, list(c)], axis=1)
                                          for c in combinations(range(Z.shape[1]), k)]))
        return np.hstack(parts)


def _ladder(orders):
    from sklearn.base import BaseEstimator, TransformerMixin

    class E(_Expand, BaseEstimator, TransformerMixin):
        pass
    return lambda: make_pipeline(E(orders), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def measure_F1(Z, y, folds):
    """Rungs: 1 = additive (splines per feature), 2 = + all pairwise products,
    3 = + all three-way products, 4 = unrestricted (best of boosted trees and RBF
    SVM).  Order = the first rung within ORDER_TOL of the best rung.  Explicit
    expansions rather than constrained trees: trees restricted to one or two
    features represent sums and products poorly, which made a tree-only ladder
    measure the trees instead of the labels."""
    p = Z.shape[1]
    rungs = {"r1_additive": _ladder(()), "r2_pairs": _ladder((2,)), "r3_triples": _ladder((2, 3))}
    if p < 3:
        rungs.pop("r3_triples")
    auc = {k: np.mean([_auc(y[te], _score(_fit(mk, Z[tr], y[tr]), Z[te])) for tr, te in folds])
           for k, mk in rungs.items()}
    auc["r4_any"] = max(np.mean([_auc(y[te], _score(_fit(lambda: mk(p), Z[tr], y[tr]), Z[te])) for tr, te in folds])
                        for mk in (lambda p: _hgb(), _svm))
    top = max(auc.values())
    if top - 0.5 < SIGNAL_MIN:
        order = "none found"
    else:
        order = next(k for k in ("r1_additive", "r2_pairs", "r3_triples", "r4_any")
                     if k in auc and auc[k] >= top - ORDER_TOL)
        order = {"r1_additive": "1", "r2_pairs": "2", "r3_triples": "3", "r4_any": ">=4"}[order]
    out = {f"F1_{k}": v for k, v in auc.items()}
    out["F1_order"] = order
    return out


# ---------------------------------------------------------------------------

def all_measures(Z, y, name, folds=None):
    """every screen measure for one dataset: inputs Z, labels y, a name (for seeds).
    folds: optional outer folds (real data grouped by patient); default = the
    students' StratifiedKFold.  L2's inner folds stay ungrouped (descriptive only)."""
    Z = np.asarray(Z, float)
    y = np.asarray(y).astype(int)
    folds = outer_folds(Z, y) if folds is None else folds
    out = {"name": name}
    out.update(measure_L1(Z, y, name, folds))
    out.update(measure_L2(Z, y, name, folds))
    out.update(measure_L3(Z, y))
    out.update(measure_F1(Z, y, folds))
    return out
