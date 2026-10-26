"""
The cells every study uses, and the coupling rule.

A "cell" is one (coupling level c, anneal time tau) setting.  The window is
c >= 2 and tau = 3-20 with full readout (docs/aqfm_overview.tex, Table 3); the
studies use a 3 x 4 grid spread across it, plus two control cells outside it.
"""

from __future__ import annotations

WINDOW_C = (2.0, 4.0, 8.0)
WINDOW_TAU = (3.0, 5.0, 10.0, 20.0)

CONTROLS = (
    (0.5, 5.0, "control: weak coupling"),
    (4.0, 1.0, "control: short anneal"),
)


def cells():
    """list of dicts {c, tau, region}: 12 window cells first, then the 2 controls"""
    out = [dict(c=c, tau=t, region="window") for c in WINDOW_C for t in WINDOW_TAU]
    out += [dict(c=c, tau=t, region=r) for c, t, r in CONTROLS]
    return out


def coupling(p: int, c: float, kind: str) -> float:
    """J for a p-qubit map at coupling level c: J = c / (p - 1), uniform all-to-all
    (the tau sweep's rule).  The J = 0 twin has no coupling at all."""
    if kind == "J0":
        return 0.0
    return float(c) / max(p - 1, 1)


def cell_name(c: float, tau: float) -> str:
    return f"c{c:g}_t{tau:g}"
