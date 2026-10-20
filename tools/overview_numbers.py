#!/usr/bin/env python3
"""
Recompute every number quoted in docs/aqfm_overview.tex from the saved results,
and check each against the value printed in the document.

Usage:
  python tools/overview_numbers.py --runs runs
Needs:
  runs/tau_sweep/ADDONS_DATASETS.csv      (written by `ts_addons.py score`)
  runs/teacher_student_v2/TS_TEACHERS.csv (optional: the 216-dataset count)
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPP = ["auc_S_MF", "auc_S_J0", "auc_S_GS", "auc_classical_best", "auc_nonlinear_best"]
C_LEVELS = [0.5, 1.0, 2.0, 4.0, 8.0]
TAUS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0]

# ---- values as printed in docs/aqfm_overview.tex ---------------------------
EXPECTED = {
    "n datasets, main run": 864,
    "n datasets, earlier run": 216,
    "oracle AUC (median, 2 dp)": 0.95,
    "region n": {"Z-only": 180, "below (weak coupling or short anneal)": 60, "transition (kappa=1)": 30,
                 "other": 27, "window": 63},
    "region median margin (2 dp)": {"Z-only": 0.00, "below (weak coupling or short anneal)": 0.00,
                                    "transition (kappa=1)": 0.03, "other": 0.04, "window": 0.15},
    "region margin>=0.05 (%)": {"Z-only": 0, "below (weak coupling or short anneal)": 0,
                                "transition (kappa=1)": 30, "other": 33, "window": 87},
    "window exact student AUC (2 dp)": 0.94,
    "window best opponent AUC (2 dp)": 0.79,
    "window share of above-chance signal (%)": 65,
    "MF teachers, window: classical/nonlinear AUC (2 dp)": 0.71,
    "MF teachers, window: mean-field student AUC (2 dp)": 0.95,
    "entanglement, tau=1, c>=2 (2 dp)": 0.98,
    "margin, tau=1, c>=2 (2 dp)": 0.00,
    "Z-only datasets with margin>=0.05": 0,
    "shots: exact student AUC (3 dp)": {"noiseless": 0.941, 10000: 0.937, 1000: 0.925, 100: 0.856},
    "shots: median margin (3 dp)": {"noiseless": 0.149, 10000: 0.144, 1000: 0.124, 100: 0.062},
    "shots: margin>=0.05 (%)": {"noiseless": 87, 10000: 87, 1000: 83, 100: 56},
    "window map (all opponents)": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 2, 1],
        [0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 3],
        [0, 0, 1, 0, 0, 1, 2, 2, 3, 3, 3, 3]],
    "window map (1000 shots)": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 1, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 3, 0],
        [0, 0, 0, 0, 0, 0, 2, 2, 3, 3, 2, 0]],
    "J table (3 dp)": {4: [0.167, 0.333, 0.667, 1.333, 2.667], 6: [0.1, 0.2, 0.4, 0.8, 1.6],
                       8: [0.071, 0.143, 0.286, 0.571, 1.143]},
    "menu sizes": {"S_Q": 540, "S_MF": 540, "S_J0": 108, "S_GS": 15},
}

results = []


def check(name, got, want):
    ok = (got == want) if not isinstance(got, float) else bool(np.isclose(got, want, atol=1e-12))
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}  (document: {want})")


def pct(v):
    return int(round(100 * float(v)))


def wmap(q, col):
    m = (q.assign(w=q[col] >= 0.05).pivot_table(index="kappa", columns="tau", values="w", aggfunc="sum")
         .reindex(index=C_LEVELS, columns=TAUS).fillna(0).astype(int))
    return m.values.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args()
    A = pd.read_csv(a.runs / "tau_sweep" / "ADDONS_DATASETS.csv")
    E = EXPECTED

    check("n datasets, main run", len(A), E["n datasets, main run"])
    v2 = a.runs / "teacher_student_v2" / "TS_TEACHERS.csv"
    if v2.exists():
        check("n datasets, earlier run", len(pd.read_csv(v2)), E["n datasets, earlier run"])
    else:
        print(f"[skip] earlier-run count: {v2} not found")
    check("oracle AUC (median, 2 dp)", round(A.oracle_auc.median(), 2), E["oracle AUC (median, 2 dp)"])

    q = A[A.kind == "Q"]
    g = q.groupby("region")
    for reg, n in E["region n"].items():
        check(f"region n [{reg}]", int(g.size()[reg]), n)
    for reg, m in E["region median margin (2 dp)"].items():
        check(f"region median margin [{reg}]", round(float(g.margin_new.median()[reg]), 2) + 0.0, m)
    for reg, f in E["region margin>=0.05 (%)"].items():
        check(f"region margin>=0.05 [{reg}]", pct((q[q.region == reg].margin_new >= 0.05).mean()), f)

    w = q[q.region == "window"]
    best = w[OPP].max(axis=1)
    check("window exact student AUC", round(w.auc_S_Q.median(), 2), E["window exact student AUC (2 dp)"])
    check("window best opponent AUC", round(best.median(), 2), E["window best opponent AUC (2 dp)"])
    check("window share of above-chance signal (%)", pct(((best - 0.5) / (w.auc_S_Q - 0.5)).median()),
          E["window share of above-chance signal (%)"])

    mf = A[(A.kind == "MF") & (A.readout == "local") & (A.region == "window")]
    check("MF teachers, window: classical", round(mf.auc_classical_best.median(), 2),
          E["MF teachers, window: classical/nonlinear AUC (2 dp)"])
    check("MF teachers, window: nonlinear", round(mf.auc_nonlinear_best.median(), 2),
          E["MF teachers, window: classical/nonlinear AUC (2 dp)"])
    check("MF teachers, window: mean-field student", round(mf.auc_S_MF.median(), 2),
          E["MF teachers, window: mean-field student AUC (2 dp)"])

    t1 = q[(q.readout == "local") & (q.kappa >= 2) & (q.tau == 1)]
    check("entanglement, tau=1, c>=2", round(t1.entanglement.median(), 2), E["entanglement, tau=1, c>=2 (2 dp)"])
    check("margin, tau=1, c>=2", round(t1.margin_new.median(), 2), E["margin, tau=1, c>=2 (2 dp)"])
    check("Z-only datasets with margin>=0.05", int((q[q.readout == "zlocal"].margin_new >= 0.05).sum()),
          E["Z-only datasets with margin>=0.05"])

    for S in ("noiseless", 10000, 1000, 100):
        auc = w.auc_S_Q if S == "noiseless" else w[f"auc_S_Q_shots{S}"]
        mar = w.margin_new if S == "noiseless" else w[f"margin_shots{S}"]
        check(f"shots {S}: exact student AUC", round(auc.median(), 3), E["shots: exact student AUC (3 dp)"][S])
        check(f"shots {S}: median margin", round(mar.median(), 3), E["shots: median margin (3 dp)"][S])
        check(f"shots {S}: margin>=0.05 (%)", pct((mar >= 0.05).mean()), E["shots: margin>=0.05 (%)"][S])

    ql = q[q.readout == "local"]
    check("window map (all opponents)", wmap(ql, "margin_new"), E["window map (all opponents)"])
    check("window map (1000 shots)", wmap(ql, "margin_shots1000"), E["window map (1000 shots)"])

    for p, row in E["J table (3 dp)"].items():
        check(f"J table p={p}", [round(c / (p - 1), 3) for c in C_LEVELS], row)

    import ts_tau_sweep as tt
    from aqfm_prescreen.aqfm import AQFM_FAMILIES
    d = tt.DESIGN
    nf, nk, nt = len(AQFM_FAMILIES), len(d["student_kappas"]), len(d["student_taus"])
    check("menu S_Q", nf * nk * nt, E["menu sizes"]["S_Q"])
    check("menu S_MF", nf * nk * nt, E["menu sizes"]["S_MF"])
    check("menu S_J0", nf * nt, E["menu sizes"]["S_J0"])
    check("menu S_GS", len(d["gs_families"]) * nk, E["menu sizes"]["S_GS"])

    print(f"\n{sum(results)}/{len(results)} checks pass")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
