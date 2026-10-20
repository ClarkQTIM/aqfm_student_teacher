"""
Additional classical comparators (protocol v2).  These are ADDED alongside the
original five (Raw, PolyInteractions, Poly2Full, Fourier, RFF); nothing is
replaced.

Fairness principle
------------------
AQFM's reported frontier picks (J, tau, observable family) by the best TEST-fold
mean (post-hoc).  The label-informed comparators below are fit on the TRAIN
fold only, so any bias in the comparison still favours AQFM.

All functions take ALREADY preprocessed inputs (fold-scaled + clipped Z).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

# ---------------------------------------------------------------------------
# 1. Feature-balanced Fourier
# ---------------------------------------------------------------------------

def balanced_fourier(Z: np.ndarray, D: int) -> np.ndarray:
    """Harmonic-major, feature-balanced fill:
        sin(1 z_1..z_p), cos(1 z_1..z_p), sin(2 z_1..z_p), cos(2 z_...), ...
    so D = p gives one function of EVERY feature (the original axis-cycling
    order gives sin+cos of only the first ceil(p/2) features at D = p)."""
    n, p = Z.shape
    cols, k = [], 1
    while len(cols) < D:
        for fn in (np.sin, np.cos):
            for j in range(p):
                if len(cols) < D:
                    cols.append(fn(k * Z[:, j]))
        k += 1
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# 2. Label-matched Fourier (Lomb-Scargle on the training fold)
# ---------------------------------------------------------------------------

OMEGA_GRID = np.linspace(0.05, 15.0, 900)   # angular frequency on standardized z


def _centered_targets(y: np.ndarray) -> np.ndarray:
    """(n, k) centred targets: binary -> 1 column; multiclass -> one-hot."""
    classes = np.unique(y)
    if len(classes) == 2:
        t = (y == classes[1]).astype(float)[:, None]
    else:
        t = (y[:, None] == classes[None, :]).astype(float)
    return t - t.mean(axis=0, keepdims=True)


def _local_maxima(power: np.ndarray, min_sep: int) -> np.ndarray:
    """Indices of local maxima, greedily thinned to be >= min_sep apart."""
    cand = np.where((power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:]))[0] + 1
    cand = cand[np.argsort(power[cand])[::-1]]
    keep = []
    for c in cand:
        if all(abs(c - k) >= min_sep for k in keep):
            keep.append(c)
    return np.array(keep, dtype=int)


class LabelMatchedFourier:
    """Fit on (Ztr, ytr):
        for each feature j, Lomb-Scargle periodogram of the centred label(s)
        against z_j; keep local maxima; rank ALL (feature, frequency) peaks
        globally by power.
    Transform with budget D:
        binary     -> one column per peak, phase fitted on train:
                      a cos(w z_j) + b sin(w z_j)   (= A cos(w z_j - phi))
        multiclass -> cos and sin per peak (2 columns), truncated to D.
    """

    def __init__(self, omegas=OMEGA_GRID, peaks_per_feature=12, min_sep=0.3):
        self.omegas = omegas
        self.ppf = peaks_per_feature
        self.min_sep_idx = max(1, int(min_sep / (omegas[1] - omegas[0])))

    def fit(self, Ztr: np.ndarray, ytr: np.ndarray):
        T = _centered_targets(np.asarray(ytr))
        self.binary = T.shape[1] == 1
        peaks = []   # (power, j, omega, a, b)
        for j in range(Ztr.shape[1]):
            z = Ztr[:, j]
            if np.ptp(z) == 0:
                continue
            power = sum(lombscargle(z, T[:, c], self.omegas) for c in range(T.shape[1]))
            for i in _local_maxima(power, self.min_sep_idx)[: self.ppf]:
                w = self.omegas[i]
                # phase fit (binary): least squares of centred label on cos/sin
                B = np.column_stack([np.cos(w * z), np.sin(w * z)])
                coef = np.linalg.lstsq(B, T[:, 0], rcond=None)[0]
                peaks.append((power[i], j, w, coef[0], coef[1]))
        peaks.sort(key=lambda r: -r[0])
        self.peaks = peaks
        return self

    def transform(self, Z: np.ndarray, D: int) -> np.ndarray | None:
        cols = []
        for _, j, w, a, b in self.peaks:
            if self.binary:
                cols.append(a * np.cos(w * Z[:, j]) + b * np.sin(w * Z[:, j]))
            else:
                cols += [np.cos(w * Z[:, j]), np.sin(w * Z[:, j])]
            if len(cols) >= D:
                break
        if len(cols) < D:          # not enough distinct peaks for this budget
            return None
        return np.column_stack(cols[:D])


# ---------------------------------------------------------------------------
# 3. Categorical handling (discrete-coded features, e.g. Tic-Tac-Toe)
# ---------------------------------------------------------------------------

MAX_LEVELS = 5


def discrete_levels(Ztr: np.ndarray):
    """Per feature: sorted unique train values if <= MAX_LEVELS, else None."""
    out = []
    for j in range(Ztr.shape[1]):
        u = np.unique(Ztr[:, j])
        out.append(u if len(u) <= MAX_LEVELS else None)
    return out


def has_discrete(Ztr: np.ndarray) -> bool:
    return any(l is not None for l in discrete_levels(Ztr))


def one_hot(Ztr, Zte):
    """Discrete features -> full one-hot (train levels); continuous -> raw z."""
    levels = discrete_levels(Ztr)
    def tf(Z):
        cols = []
        for j, lv in enumerate(levels):
            if lv is None:
                cols.append(Z[:, j])
            else:
                cols += [np.isclose(Z[:, j], v).astype(float) for v in lv]
        return np.column_stack(cols)
    return tf(Ztr), tf(Zte)


def level_indicator(Ztr, Zte, ytr):
    """D = p.  Discrete feature -> indicator of the single level whose train
    label mean deviates most from the overall mean (binary: P(y=1)); continuous
    feature -> raw z.  Label-informed, train fold only."""
    levels = discrete_levels(Ztr)
    T = _centered_targets(np.asarray(ytr))
    chosen = []
    for j, lv in enumerate(levels):
        if lv is None:
            chosen.append(None)
            continue
        dev = [np.abs(T[np.isclose(Ztr[:, j], v)].mean(axis=0)).sum() for v in lv]
        chosen.append(lv[int(np.argmax(dev))])
    def tf(Z):
        return np.column_stack([Z[:, j] if v is None else
                                np.isclose(Z[:, j], v).astype(float)
                                for j, v in enumerate(chosen)])
    return tf(Ztr), tf(Zte)


# ---------------------------------------------------------------------------
# 4. ANOVA-matched classical control (same order structure as AQFM families)
# ---------------------------------------------------------------------------
#
# AQFM at J=0:  local family  = per-feature nonlinear functions   (D = p, 3p)
#               pair family   = products of per-feature functions (D = q, 3q)
#               full          = local + pair                      (D = 3p + 3q)
# This control uses probabilists' Hermite polynomials He_1..He_3 as the
# per-feature functions and their same-degree products for pairs, at exactly
# AQFM's natural dimensions.  If the "structure -> order" result is a property
# of observable ORDER rather than of AQFM specifically, this family should
# reproduce it.

def _hermite(z, k):
    return (z, z * z - 1.0, z ** 3 - 3.0 * z)[k - 1]


def anova_matched(Z: np.ndarray, kind: str) -> np.ndarray:
    """kind in {'L1' (D=p), 'L3' (D=3p), 'P1' (D=q), 'P3' (D=3q), 'F3' (D=3p+3q)}."""
    n, p = Z.shape
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    def local(K):
        return [_hermite(Z[:, j], k) for k in range(1, K + 1) for j in range(p)]
    def pair(K):
        return [_hermite(Z[:, i], k) * _hermite(Z[:, j], k)
                for k in range(1, K + 1) for i, j in pairs]
    cols = {"L1": local(1), "L3": local(3), "P1": pair(1), "P3": pair(3),
            "F3": local(3) + pair(3)}[kind]
    return np.column_stack(cols) if cols else np.empty((n, 0))


ANOVA_KINDS = ("L1", "L3", "P1", "P3", "F3")
