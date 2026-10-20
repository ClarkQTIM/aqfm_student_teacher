"""
Dial experiment: when does AQFM beat a TUNABLE classical basis at fixed D,
and can cheap pre-run quantities predict it?

Labels
------
    latent = alpha * A + (1 - alpha) * C + sigma * eps,   y = 1[latent > median]
A = an AQFM-shaped component, C = a classical-shaped component (both
standardised).  alpha = 0 -> purely classical target, alpha = 1 -> the label
is (a noisy threshold of) an AQFM observable.

Families (p = 6):
    local      A = sum_{j<3} X_j(z_j; J=0, tau=2)          (1-D chirped curves)
               C = sum_{j<3} sin(1.3 z_j)   (plain odd sinusoids; A is even)
    pair       A = sum_k X(z_2k; J=0,tau=2) X(z_2k+1; J=0,tau=2)   (curve products)
               C = sum_k z_2k z_2k+1                          (plain products)
    collective A = sum_i <X_i>(z; J=1, tau=5)                (uniform-coupling,
               C = sum_k z_2k z_2k+1                           beyond-pairwise)

A is computed on globally standardised + clipped inputs, the same transform
the per-fold pipeline applies (up to fold-to-fold scaler noise).

Gap (the outcome)
-----------------
For budgets D in {p, 3p, 3p+3q}: nested CV (outer 5-fold, inner 3-fold on the
outer-train rows) picks the best AQFM configuration (family, J, tau) with
dim <= D and, separately, the best TUNABLE classical candidate with dim <= D.
gap = outer AUC(AQFM) - outer AUC(classical).  Also reported with AQFM
restricted to J = 0 (a classically simulable dictionary).

Predictors (computed on an INDEPENDENT screening sample from the same
generator, before any gap is computed)
---------------------------------------------------------------------
    P1_J0proxy_D : CV AUC(best J=0 AQFM, dim<=D) - CV AUC(best classical, dim<=D)
                   J=0 AQFM is a product state => exactly simulable at any p.
    P2_conc      : share of the label's Lomb-Scargle power (summed over
                   features) in the top-p peaks.  High => a few sinusoids
                   explain the label => classical Fourier should be competitive.
    P3_beyond    : M3 - M2 of the structural screen (HGB minus additive+
                   pairwise splines): signal that is neither additive nor
                   pairwise, i.e. where a collective map could have room.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import lombscargle
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from .aqfm import (AQFM_FAMILIES, compute_bank, family_matrix, single_qubit_strang,
                   slices_for_p, strang_steps)
from .dictionaries import (OMEGA_GRID, LabelMatchedFourier, _centered_targets,
                           _local_maxima, anova_matched, balanced_fourier)
from .evaluate import INPUT_CLIP, TAU_GRID, fold_splits, preprocess_fold_inputs

P_DIAL = 6
N_DIAL = 600
SIGMA = 0.25
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FAMILIES = ("local", "pair", "collective", "triple", "radial")
# "triple" and "radial" are NEGATIVE CONTROLS added after the first run:
# beyond-pairwise targets that are NOT built from AQFM observables, to measure
# how often the frozen P3 rule fires when AQFM has no special fit.
J_GRID_DIAL = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def _global_z(X):
    return np.clip((X - X.mean(0)) / X.std(0), -INPUT_CLIP, INPUT_CLIP)


def _curve(z, tau, k):
    """J=0 single-qubit response (k: 0=X, 1=Y, 2=Z) at the protocol's step rule."""
    return single_qubit_strang(z, tau, strang_steps(tau))[:, k]


def components(family, Zg):
    """Return (A, C) un-standardised for globally standardised inputs Zg."""
    n, p = Zg.shape
    if family == "local":
        A = sum(_curve(Zg[:, j], 2.0, 0) for j in range(3))
        C = sum(np.sin(1.3 * Zg[:, j]) for j in range(3))
    elif family == "pair":
        A = sum(_curve(Zg[:, 2 * k], 2.0, 0) * _curve(Zg[:, 2 * k + 1], 2.0, 0)
                for k in range(p // 2))
        C = sum(Zg[:, 2 * k] * Zg[:, 2 * k + 1] for k in range(p // 2))
    elif family == "collective":
        bank = compute_bank(Zg, 1.0, 5.0, method="strang").astype(float)
        A = family_matrix(bank, "X", p).sum(1)
        C = sum(Zg[:, 2 * k] * Zg[:, 2 * k + 1] for k in range(p // 2))
    elif family == "triple":          # smooth 3-way products (non-AQFM)
        A = Zg[:, 0] * Zg[:, 1] * Zg[:, 2] + Zg[:, 3] * Zg[:, 4] * Zg[:, 5]
        C = sum(Zg[:, 2 * k] * Zg[:, 2 * k + 1] for k in range(p // 2))
    elif family == "radial":          # permutation-symmetric, collective (non-AQFM)
        A = np.cos(1.5 * np.linalg.norm(Zg, axis=1))
        C = sum(Zg[:, 2 * k] * Zg[:, 2 * k + 1] for k in range(p // 2))
    else:
        raise ValueError(family)
    return A, C


def generate(family, alpha, seed, n=N_DIAL, p=P_DIAL, sigma=SIGMA):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    A, C = components(family, _global_z(X))
    latent = alpha * _std(A) + (1 - alpha) * _std(C) + sigma * rng.standard_normal(n)
    y = (latent > np.median(latent)).astype(int)
    return X, y


def dial_name(family, alpha, rep):
    return f"dial_{family}_a{alpha:.2f}_r{rep}"


def eval_seed(family, alpha, rep):
    return 1000 * FAMILIES.index(family) + 100 * int(round(alpha * 4)) + rep


def screen_seed(family, alpha, rep):
    return 500_000 + eval_seed(family, alpha, rep)      # independent draw


# ---------------------------------------------------------------------------
# Tunable classical candidates: f(Ztr, ytr, Zte) -> (Ftr, Fte) or None
# ---------------------------------------------------------------------------

def _compress_classes(W, blocks_tr):
    """Multiclass: W is (K, d).  For each block (column slice) the per-class
    contributions on train are an (n, K) matrix; keep its leading right
    singular direction so every component is still ONE column.  Binary
    (K = 1): returns W[0] per block unchanged."""
    if W.shape[0] == 1:
        return [W[0, sl] for sl in blocks_tr[1]]
    B, slices = blocks_tr
    out = []
    for sl in slices:
        C = B[:, sl] @ W[:, sl].T                    # (n, K)
        v = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
        out.append(W[:, sl].T @ v)                   # (block,)
    return out


def _gam_shapes(Ztr, ytr, n_knots=8):
    """Label-tuned per-feature shape functions: fit an additive spline LR on
    train, return a transformer giving f_j(z_j) (one column per feature).
    Multiclass: per-feature contributions compressed to one column."""
    st = SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                           extrapolation="linear", include_bias=False).fit(Ztr)
    Btr = st.transform(Ztr)
    sc = StandardScaler().fit(Btr)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Btr), ytr)
    k = Btr.shape[1] // Ztr.shape[1]
    slices = [slice(j * k, (j + 1) * k) for j in range(Ztr.shape[1])]
    ws = _compress_classes(m.coef_, (sc.transform(Btr), slices))

    def tf(Z):
        B = sc.transform(st.transform(Z))
        return np.column_stack([B[:, sl] @ w for sl, w in zip(slices, ws)])
    return tf


def _pair_products(F):
    p = F.shape[1]
    return np.column_stack([F[:, i] * F[:, j] for i, j in combinations(range(p), 2)])


def _ga2m(Ztr, ytr, n_knots=6, C=0.1):
    """Label-tuned GA2M components (Lou et al. 2013 style, fit in one shot):
    one joint LR on [per-feature spline blocks, all pairwise tensor-product
    blocks]; each feature's and each pair's fitted component becomes ONE
    column.  The classical analogue of J=0 AQFM 'full' (fixed 1-D shapes and
    their products), except that here the shapes are learned from labels."""
    p = Ztr.shape[1]
    st = SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                           extrapolation="linear", include_bias=False).fit(Ztr)
    k = st.transform(Ztr[:1]).shape[1] // p
    pairs = list(combinations(range(p), 2))

    def design(Z):
        B = st.transform(Z)
        blocks = [B[:, j * k:(j + 1) * k] for j in range(p)]
        P = [np.einsum("na,nb->nab", blocks[i], blocks[j]).reshape(len(Z), -1)
             for i, j in pairs]
        return np.hstack([B] + P)
    Dtr = design(Ztr)
    sc = StandardScaler().fit(Dtr)
    W = LogisticRegression(C=C, max_iter=10_000).fit(sc.transform(Dtr), ytr).coef_
    edges = np.cumsum([0] + [k] * p + [k * k] * len(pairs))
    slices = [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
    ws = _compress_classes(W, (sc.transform(Dtr), slices))

    def tf(Z):
        A = sc.transform(design(Z))
        comps = np.column_stack([A[:, sl] @ w for sl, w in zip(slices, ws)])
        return comps[:, :p], comps[:, p:]
    return tf


def classical_candidates(p, opponents="v1"):
    """[(name, D, fn)] — fn(Ztr, ytr, Zte) -> (Ftr, Fte) | None.
    opponents='v1': the frozen first-run set.
    opponents='v2': v1 + GA2M pair/full components (label-tuned pair shapes)."""
    q = p * (p - 1) // 2
    C = []

    def fixed(name, D, f):
        C.append((name, D, lambda A, y, B, f=f: (f(A), f(B))))

    fixed("Raw", p, lambda Z: Z)
    for D in (p, 3 * p, 3 * p + 3 * q):
        fixed(f"FourierBalanced{D}", D, lambda Z, D=D: balanced_fourier(Z, D))
        fixed(f"RFF{D}", D, lambda Z, D=D: RBFSampler(gamma=1.0 / p, n_components=D,
                                                       random_state=42).fit(np.zeros((1, p))).transform(Z))
    for kind in ("L3", "P1", "P3", "F3"):
        D = {"L3": 3 * p, "P1": q, "P3": 3 * q, "F3": 3 * p + 3 * q}[kind]
        fixed(f"Hermite{kind}", D, lambda Z, k=kind: anova_matched(Z, k))

    def spline3(A, y, B):
        st = SplineTransformer(n_knots=2, degree=2, knots="quantile",
                               extrapolation="linear", include_bias=True).fit(A)
        return st.transform(A), st.transform(B)
    C.append(("Spline3", 3 * p, spline3))

    def poly_int(A, y, B):
        pf = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(A)
        return pf.transform(A), pf.transform(B)
    C.append(("PolyInteractions", p + q, poly_int))

    for D in (p, 3 * p):
        def lmf(A, y, B, D=D):
            m = LabelMatchedFourier().fit(A, y)
            Fa, Fb = m.transform(A, D), m.transform(B, D)
            return None if Fa is None else (Fa, Fb)
        C.append((f"LabelFourier{D}", D, lmf))

    def shapes(A, y, B):
        tf = _gam_shapes(A, y)
        return tf(A), tf(B)
    C.append(("GAMShapes", p, shapes))

    def shape_products(A, y, B):
        tf = _gam_shapes(A, y)
        return _pair_products(tf(A)), _pair_products(tf(B))
    C.append(("ShapeProducts", q, shape_products))

    def shape_full(A, y, B):
        tf = _gam_shapes(A, y)
        Fa, Fb = tf(A), tf(B)
        return np.hstack([Fa, _pair_products(Fa)]), np.hstack([Fb, _pair_products(Fb)])
    C.append(("ShapeFull", p + q, shape_full))

    if opponents in ("v2", "v2sym"):
        def ga2m_pairs(A, y, B):
            tf = _ga2m(A, y)
            return tf(A)[1], tf(B)[1]
        C.append(("GA2MPairs", q, ga2m_pairs))

        def ga2m_full(A, y, B):
            tf = _ga2m(A, y)
            return np.hstack(tf(A)), np.hstack(tf(B))
        C.append(("GA2MFull", p + q, ga2m_full))
    elif opponents != "v1":
        raise ValueError(opponents)

    if opponents == "v2sym":
        # Permutation-symmetric aggregates (added 2026-10-08 after the dial's
        # radial control showed they beat AQFM there): spline bases of
        # mean(z) and mean(z^2), alone and appended to the raw features.
        k = max(2, p // 2)

        def _sym(A, B):
            agg = lambda Z: np.column_stack([Z.mean(1), (Z ** 2).mean(1)])
            st = SplineTransformer(n_knots=k, degree=1, knots="quantile",
                                   extrapolation="linear", include_bias=True).fit(agg(A))
            return st.transform(agg(A)), st.transform(agg(B))

        def sym_only(A, y, B):
            return _sym(A, B)
        C.append(("SymAggSplines", 2 * (k + 0), sym_only))

        def sym_raw(A, y, B):
            a, b = _sym(A, B)
            return np.hstack([A, a]), np.hstack([B, b])
        C.append(("RawPlusSymAgg", p + 2 * k, sym_raw))
    return C


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def lr_auc(Ftr, Fte, ytr, yte):
    """Binary: ROC-AUC.  Multiclass: macro one-vs-rest AUC over the classes
    seen in training (same convention as evaluate.fit_eval_lr)."""
    sc = StandardScaler().fit(Ftr)
    m = LogisticRegression(C=1.0, max_iter=10_000).fit(sc.transform(Ftr), ytr)
    P = m.predict_proba(sc.transform(Fte))
    if len(m.classes_) == 2:
        return roc_auc_score(yte, P[:, 1])
    # macro one-vs-rest over classes that have both positives and negatives in
    # this test fold (small classes can be absent from a fold)
    aucs = [roc_auc_score(yte == c, P[:, i]) for i, c in enumerate(m.classes_)
            if 0 < (yte == c).sum() < len(yte)]
    return float(np.mean(aucs))


def budgets(p):
    q = p * (p - 1) // 2
    return (p, 3 * p, 3 * p + 3 * q)


def aqfm_candidates(p, Js=J_GRID_DIAL, taus=TAU_GRID):
    sl, _ = slices_for_p(p)
    dims = {"X": p, "Y": p, "Z": p, "XYZ": 3 * p, "XX": sl["XX"].stop - sl["XX"].start}
    q = dims["XX"]
    dims.update({"YY": q, "ZZ": q, "pairs": 3 * q, "full": 3 * p + 3 * q})
    return [(fam, J, tau, dims[fam]) for J in Js for tau in taus for fam in AQFM_FAMILIES]


def fold_banks(X, y, Js=J_GRID_DIAL, taus=TAU_GRID, mf_Js=()):
    """{(fold, J, tau): exact bank} and {(fold, "MF", J, tau): mean-field bank},
    on fold-scaled inputs (rows aligned with X)."""
    out = {}
    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        Z = np.empty_like(X, dtype=float); Z[tr], Z[te] = Ztr, Zte
        for J in Js:
            for tau in taus:
                out[(f, J, tau)] = compute_bank(Z, J, tau, method="strang").astype(float)
        for J in mf_Js:
            for tau in taus:
                out[(f, "MF", J, tau)] = compute_bank(Z, J, tau, method="meanfield").astype(float)
    return out


# Opponent set v3 = v2 classical + the classically simulable versions of the
# SAME map (rung 4): exact J=0 banks and mean-field J=1 banks.  The question
# answered is then "does exact J!=0 beat everything classical, including its
# own classical twins?"
SIMULABLE_SOURCES = (("J0", 0.0), ("MF", 1.0))


def _bank_key(f, src, J, tau):
    return (f, J, tau) if src == "J0" else (f, "MF", J, tau)


# ---------------------------------------------------------------------------
# Outcome: nested fixed-D gap
# ---------------------------------------------------------------------------

def nested_gap(name, X, y, banks=None, inner_splits=3, seed=42, opponents="v1",
               Js=J_GRID_DIAL, taus=TAU_GRID, mf_Js=None):
    """Js: exact-AQFM couplings (must include 0.0).  mf_Js: couplings whose
    mean-field twin joins the classical pool (v3; default = nonzero Js)."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    p = X.shape[1]
    v3 = opponents in ("v3", "v4")
    base = {"v3": "v2", "v4": "v2sym"}.get(opponents, opponents)
    Js = tuple(Js)
    mf_Js = tuple(J for J in Js if J != 0) if mf_Js is None else tuple(mf_Js)
    if banks is None:
        banks = fold_banks(X, y, Js=Js, taus=taus, mf_Js=mf_Js if v3 else ())
    ccands = classical_candidates(p, base)
    acands = aqfm_candidates(p, Js=Js, taus=taus)
    sources = [("J0", 0.0)] + [("MF", J) for J in mf_Js]
    scands = ([(src, fam, J, tau, D) for src, J in sources
               for fam, _, tau, D in aqfm_candidates(p, Js=(J,), taus=taus)] if v3 else [])
    rows = []
    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        ytr, yte = y[tr], y[te]
        inner = list(StratifiedKFold(inner_splits, shuffle=True,
                                     random_state=seed).split(Ztr, ytr))
        # inner scores -------------------------------------------------------
        a_inner = {}
        for fam, J, tau, D in acands:
            F = family_matrix(banks[(f, J, tau)], fam, p)[tr]
            a_inner[(fam, J, tau, D)] = np.mean(
                [lr_auc(F[i], F[v], ytr[i], ytr[v]) for i, v in inner])
        c_inner = {}
        for nm, D, fn in ccands:
            s = []
            for i, v in inner:
                out = fn(Ztr[i], ytr[i], Ztr[v])
                if out is None:
                    break
                s.append(lr_auc(out[0], out[1], ytr[i], ytr[v]))
            if len(s) == len(inner):
                c_inner[(nm, D)] = np.mean(s)
        for src, fam, J, tau, D in scands:                 # rung-4 opponents
            if src == "J0":
                c_inner[(f"J0:{fam}|tau{tau:g}", D)] = a_inner[(fam, J, tau, D)]
            else:
                F = family_matrix(banks[_bank_key(f, src, J, tau)], fam, p)[tr]
                c_inner[(f"MF:{fam}|J{J:g}|tau{tau:g}", D)] = np.mean(
                    [lr_auc(F[i], F[v], ytr[i], ytr[v]) for i, v in inner])
        sim_lookup = {(f"J0:{fam}|tau{tau:g}" if src == "J0" else
                       f"MF:{fam}|J{J:g}|tau{tau:g}"): (src, fam, J, tau)
                      for src, fam, J, tau, D in scands}

        def outer_aqfm(key):
            fam, J, tau, _ = key
            F = family_matrix(banks[(f, J, tau)], fam, p)
            return lr_auc(F[tr], F[te], ytr, yte)

        def outer_classical(key):
            if key[0] in sim_lookup:
                src, fam, J, tau = sim_lookup[key[0]]
                F = family_matrix(banks[_bank_key(f, src, J, tau)], fam, p)
                return lr_auc(F[tr], F[te], ytr, yte)
            fn = {nm: fn for nm, _, fn in ccands}[key[0]]
            Fa, Fb = fn(Ztr, ytr, Zte)
            return lr_auc(Fa, Fb, ytr, yte)

        for B in budgets(p):
            ca = {k: v for k, v in c_inner.items() if k[1] <= B}
            ck = max(ca, key=ca.get)
            c_out = outer_classical(ck)
            extra = {}
            if v3:      # the twin alone: best mean-field config within budget
                mfa = {k: v for k, v in ca.items() if k[0].startswith("MF:")}
                mk = max(mfa, key=mfa.get)
                extra = {"mf_choice": mk[0], "mf_outer": outer_classical(mk)}
            for scope, Jsc in (("all", Js), ("J0", (0.0,))):
                aa = {k: v for k, v in a_inner.items() if k[3] <= B and k[1] in Jsc}
                ak = max(aa, key=aa.get)
                a_out = outer_aqfm(ak)
                rows.append({"dataset": name, "fold": f, "budget": B, "aqfm_scope": scope,
                             "aqfm_choice": f"{ak[0]}|J{ak[1]:g}|tau{ak[2]:g}",
                             "aqfm_D": ak[3], "aqfm_inner": aa[ak], "aqfm_outer": a_out,
                             "classical_choice": ck[0], "classical_D": ck[1],
                             "classical_inner": ca[ck], "classical_outer": c_out,
                             "gap": a_out - c_out, **extra,
                             **({"gap_vs_mf": a_out - extra["mf_outer"]} if v3 else {})})
    return rows


# ---------------------------------------------------------------------------
# Predictors (screening sample)
# ---------------------------------------------------------------------------

def spectrum_concentration(Z, y, top=None):
    """Sum of the top-`top` Lomb-Scargle peak heights / sum of ALL peak
    heights (local maxima, every feature).  In (0, 1]; near 1 => a few
    sinusoids carry the label."""
    T = _centered_targets(np.asarray(y))[:, 0]
    top = Z.shape[1] if top is None else top
    peaks = []
    for j in range(Z.shape[1]):
        if np.ptp(Z[:, j]) == 0:
            continue
        pw = lombscargle(Z[:, j], T, OMEGA_GRID)
        peaks += list(pw[_local_maxima(pw, 20)])
    peaks = np.sort(peaks)[::-1]
    return float(peaks[:top].sum() / (peaks.sum() + 1e-12))


def predictors(X, y, screen_fn, opponents="v1", banks=None, Js=J_GRID_DIAL,
               taus=TAU_GRID):
    """All predictors on (X, y) = the screening sample.  5-fold CV, best-of
    selection (no nesting needed: the sample is independent of the outcome).
    `banks` may be precomputed (same keys as fold_banks)."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    p = X.shape[1]
    v3 = opponents in ("v3", "v4")
    base = {"v3": "v2", "v4": "v2sym"}.get(opponents, opponents)
    nz = tuple(J for J in Js if J != 0)
    if banks is None:
        banks = fold_banks(X, y, Js=(0.0,) + nz if v3 else (0.0,), taus=taus,
                           mf_Js=nz if v3 else ())
    ccands = classical_candidates(p, base)
    acands = aqfm_candidates(p, Js=(0.0,), taus=taus)
    a_sc = {k: [] for k in acands}; c_sc = {(nm, D): [] for nm, D, _ in ccands}
    conc = []
    for f, (tr, te) in enumerate(fold_splits(X, y), 1):
        Ztr, Zte = preprocess_fold_inputs(X, tr, te)
        for key in acands:
            F = family_matrix(banks[(f, key[1], key[2])], key[0], p)
            a_sc[key].append(lr_auc(F[tr], F[te], y[tr], y[te]))
        for nm, D, fn in ccands:
            out = fn(Ztr, y[tr], Zte)
            c_sc[(nm, D)].append(np.nan if out is None else
                                 lr_auc(out[0], out[1], y[tr], y[te]))
        conc.append(spectrum_concentration(Ztr, y[tr]))
    a_m = {k: np.mean(v) for k, v in a_sc.items()}
    c_m = {k: np.nanmean(v) for k, v in c_sc.items() if not np.all(np.isnan(v))}
    out = {}
    for B in budgets(p):
        ab = max(v for k, v in a_m.items() if k[3] <= B)
        cb = max(v for k, v in c_m.items() if k[1] <= B)
        out[f"P1_J0proxy_D{B}"] = ab - cb
        out[f"J0best_D{B}"] = ab
        out[f"classbest_D{B}"] = cb
    if v3:
        # PILOT (P4): exact J=1 vs everything classical incl. its twins
        # (J=0 exact, J=1 mean-field) on the screening sample.
        def cv(bkey, fam):
            return np.mean([lr_auc(family_matrix(banks[(f,) + bkey], fam, p)[tr],
                                   family_matrix(banks[(f,) + bkey], fam, p)[te],
                                   y[tr], y[te])
                            for f, (tr, te) in enumerate(fold_splits(X, y), 1)])
        ex_t, mf = {}, {}
        for fam, J, tau, D in aqfm_candidates(p, Js=nz, taus=taus):
            ex_t[(fam, D)] = max(ex_t.get((fam, D), -1), cv((J, tau), fam))
            mf[(fam, D)] = max(mf.get((fam, D), -1), cv(("MF", J, tau), fam))
        for B in budgets(p):
            eb = max(v for k, v in ex_t.items() if k[1] <= B)
            sb = max([out[f"classbest_D{B}"], out[f"J0best_D{B}"]] +
                     [v for k, v in mf.items() if k[1] <= B])
            out[f"P4_pilot_D{B}"] = eb - sb
            out[f"P4_exactJ1_vs_MF_D{B}"] = eb - max(v for k, v in mf.items() if k[1] <= B)
    out["P2_conc"] = float(np.mean(conc))
    s = screen_fn(X, y)
    out.update({"M0": s["M0"], "M1": s["M1"], "M2": s["M2"], "M3": s["M3"],
                "Delta_NL": s["Delta_NL"], "Delta_INT": s["Delta_INT"],
                "P3_beyond": s["M3"] - s["M2"]})
    return out
