#!/usr/bin/env python3
"""
Scaling test: does the "exact map needed" window survive as the number of
qubits p grows?

Motivation (tau sweep, 2026-10-14): for labels made by the exact anneal, local
(X,Y,Z) readout, coupling J(p-1) >~ 1.25 and tau >~ 3, the exact student beats
every classical opponent (incl. the mean-field twin) by 0.1-0.3 AUC at p = 4-8.
But with uniform all-to-all coupling J = kappa/(p-1), mean-field becomes exact
as p -> infinity (each pair coupling ~ 1/p), so the window could be a
small-size effect.  This run measures the gap as a function of p.

Same machinery as ts_tau_sweep.py (same map, simulators, students, classical
opponents, nested CV, tuned C), with:
  p        4, 6, 8, 10, 12   (14 optional: --ps 14 in a second make/run)
  kappa    0.5, 1, 1.5, 2, 4  (1.5 just above the random-field threshold sqrt(pi/2) = 1.25)
  tau      1 (control), 3, 5, 7, 10   (the physically meaningful window)
  readouts local: Q teachers x 3 seeds + MF teachers x 1 seed
           zlocal: Q teachers x 1 seed (control: expected null)
  students tau grid = teacher tau grid; kappa grid = teacher kappa grid
  inputs   one shared X per p (identical to the tau sweep for p = 4, 6, 8)
  classical curves up to D = 64 p (v2 / tau sweep went to 128 p; dropped for cost at p = 12)

Phases: make / run / score (as ts_tau_sweep.py); outputs SC_*.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_student as ts            # noqa: E402
import ts_tau_sweep as tt               # noqa: E402

TAUS = [1.0, 3.0, 5.0, 7.0, 10.0]
KAPPAS = [0.5, 1.0, 1.5, 2.0, 4.0]

# Applied at import time, so worker processes (fork or spawn) see the same design.
tt.DESIGN.update({
    "ps": [4, 6, 8, 10, 12],
    "kappas": KAPPAS,
    "teacher_taus": TAUS,
    "student_kappas": KAPPAS,
    "student_taus": TAUS,
    "curve_mults": [1, 2, 4, 8, 16, 32, 64],
    "readouts": ["local", "zlocal"],
    "seeds_Q_local": 3,
    "seeds_MF_local": 1,
    "seeds_Q_zlocal": 1,
    "version": "scaling_1",
})
DESIGN = tt.DESIGN

HYPOTHESES = {
    "S1_scaling": ("Q teachers, local readout, kappa in {2, 4}, tau in {3, 5, 7, 10}: regress the margin "
                   "Delta = S_Q - max(S_MF, S_J0, S_GS, classical) on p (dataset level, bootstrap over "
                   "datasets, 2000 resamples). Decision rule: 95% CI of the slope entirely below -0.005 per "
                   "qubit -> 'shrinks'; entirely above +0.005 -> 'grows'; otherwise 'flat/undetermined'. "
                   "No direction is predicted (mean-field exactness at large p argues 'shrinks'; "
                   "structural failure on balanced samples argues 'flat')."),
    "S2_threshold": ("Q teachers, local, tau in {3..10}: the fraction needing the exact map (S_Q - S_MF >= 0.01) "
                     "at kappa = 1 decreases from p = 4 to p = 12, and at p = 12 it is lower than at kappa = 1.5"),
    "S3_z_null": "Q teachers, zlocal: fraction with Delta >= 0.01 is <= 10% at every p",
    "S4_classical_cost": ("Q teachers, local, kappa in {2, 4}, tau in {3..10}: the fraction 'unreachable' by the "
                          "classical curves (within 64p features, 0.01 tolerance) does not decrease with p "
                          "(Spearman(p, unreachable indicator) >= 0)"),
    "R_report": ("report median Delta and S_Q - S_MF by (p, kappa, tau); teacher entanglement and its "
                 "concentration on balanced samples (Spearman of per-sample E with |mean x|); mf_agreement vs Delta"),
}


def teacher_specs(d=DESIGN):
    specs = []
    for p, tau, kap in product(d["ps"], d["teacher_taus"], d["kappas"]):
        for s in range(d["seeds_Q_local"]):
            specs.append(dict(kind="Q", p=p, kappa=kap, tau=tau, readout="local", seed=s))
        for s in range(d["seeds_MF_local"]):
            specs.append(dict(kind="MF", p=p, kappa=kap, tau=tau, readout="local", seed=s))
        for s in range(d["seeds_Q_zlocal"]):
            specs.append(dict(kind="Q", p=p, kappa=kap, tau=tau, readout="zlocal", seed=s))
    for sp in specs:
        sp["name"] = f"sc_{sp['kind']}_p{sp['p']}_k{sp['kappa']:g}_t{sp['tau']:g}_{sp['readout']}_s{sp['seed']}"
        sp["data_seed"] = tt._seed(sp["name"])
    return specs


# wrappers defined here so spawned workers import this module (and its design)
def _bank_job(args):
    return tt._bank_job(args)


def _run_job(args):
    return tt._job(args)


def _balanced_rho(out, p, kappa, tau):
    """Spearman over samples of per-sample entanglement vs |mean input| (exact bank)"""
    from aqfm_prescreen.aqfm import family_matrix
    B = tt.load_bank(out, p, "Q", kappa, tau)
    r2 = sum(family_matrix(B, f, p) ** 2 for f in "XYZ")
    E = (1 - r2).mean(1)
    _, Z = tt.inputs(p)
    return float(spearmanr(E, np.abs(Z.mean(1))).correlation)


def do_make(out: Path, workers: int, d=DESIGN):
    out.mkdir(parents=True, exist_ok=True)
    pf = out / "SC_PROTOCOL.json"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if pf.exists():                         # a later make (e.g. --ps 14) appends, never overwrites
        proto = json.loads(pf.read_text())
        proto.setdefault("additions", []).append({"ps": d["ps"], "written_at": stamp})
    else:
        proto = {"design": d, "hypotheses": HYPOTHESES, "written_at": stamp,
                 "note": "written by `make`, before any student was run; follows the tau sweep (TT_PROTOCOL.json)"}
    pf.write_text(json.dumps(proto, indent=2))
    for p in d["ps"]:
        (out / "banks" / f"p{p}").mkdir(parents=True, exist_ok=True)
    specs = tt.bank_specs(d)
    specs.sort(key=lambda s: -(s[0] * 100 + s[3] * (s[1] == "Q")))      # slow exact banks first
    print(f"{len(specs)} banks for p = {d['ps']}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_bank_job, (out,) + s): s for s in specs}
        for i, fu in enumerate(as_completed(futs), 1):
            nm, st = fu.result()
            if i % 25 == 0 or i == len(specs) or st.startswith("ok") and float(st.split()[1][:-1]) > 120:
                print(f"[bank {i}/{len(specs)} {time.time() - t0:6.0f}s] {st} p{futs[fu][0]} {nm}", flush=True)
    (out / "labels").mkdir(exist_ok=True)
    mf = out / "SC_TEACHERS.csv"
    old = pd.read_csv(mf) if mf.exists() else pd.DataFrame()
    have = set(old.name) if len(old) else set()
    metas = []
    for sp in teacher_specs(d):
        if sp["name"] in have:
            continue
        y, meta = tt.make_labels(out, sp, d)
        if y is not None:
            np.save(out / "labels" / f"{sp['name']}.npy", y)
        metas.append(meta)
    M = pd.concat([old, pd.DataFrame(metas)], ignore_index=True)
    M.to_csv(mf, index=False)
    ok = M[~M.degenerate.astype(bool)]
    print(f"{len(ok)} datasets in SC_TEACHERS.csv ({int(M.degenerate.astype(bool).sum())} degenerate, skipped)")
    pd.set_option("display.width", 200)
    q = ok[(ok.kind == "Q") & (ok.readout == "local")]
    print("Q teachers, local: median entanglement / mf_agreement by p x kappa (tau 3-10)")
    w = q[q.tau >= 3]
    print(w.pivot_table(index="p", columns="kappa", values="entanglement", aggfunc="median").round(2).to_string())
    print(w.pivot_table(index="p", columns="kappa", values="mf_agreement", aggfunc="median").round(2).to_string())


def do_run(out: Path, workers: int, only=None):
    M = pd.read_csv(out / "SC_TEACHERS.csv")
    M = M[~M.degenerate.astype(bool)]
    jobs = [(out, n, int(p)) for n, p in zip(M.name, M.p) if not only or any(o in n for o in only)]
    jobs.sort(key=lambda j: (-j[2], j[1]))              # biggest p first; same p together (bank cache)
    print(f"{len(jobs)} datasets", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_run_job, j): j[1] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nm, st = fu.result()
                print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{i}/{len(jobs)}] FAILED {futs[fu]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun to retry" if fails else ""))
    sys.exit(1 if fails else 0)


def _rows(out, M, d=DESIGN):
    rows = []
    for nm, kind in zip(M.name, M.kind):
        f = out / "runs" / f"{nm}.csv"
        if not f.exists():
            continue
        R = pd.read_csv(f)
        Cr = pd.read_csv(out / "runs" / f"{nm}_curve.csv")
        m = R.pivot_table(index="fold", columns="arm", values="auc").mean()
        cm = Cr.groupby(["family", "D"]).auc.mean().reset_index()
        tgt = max(m[a] for a in tt.ARMS)
        opp = max(m["S_MF"], m["S_J0"], m["S_GS"], m["classical_best"])
        rows.append(dict(name=nm, **{f"auc_{k}": v for k, v in m.items()},
                         delta=m["S_Q"] - opp, gap_Q_MF=m["S_Q"] - m["S_MF"],
                         best_student_auc=tgt, best_student=max(tt.ARMS, key=lambda k: m[k]),
                         shortfall_classical=tgt - m["classical_best"],
                         dstar=ts._dstar(cm, tgt, d["dstar_tol"])))
    return pd.DataFrame(rows)


def do_score(out: Path, d=DESIGN):
    M = pd.read_csv(out / "SC_TEACHERS.csv")
    M = M[~M.degenerate.astype(bool)]
    T = M.merge(_rows(out, M, d), on="name")
    T["dstar_over_p"] = T.dstar / T.p
    T["unreachable"] = np.isinf(T.dstar)
    T.to_csv(out / "SC_DATASETS.csv", index=False)
    cur = T.groupby(["kind", "readout", "p", "kappa", "tau"])[
        ["entanglement", "mf_agreement", "auc_S_J0", "auc_S_MF", "auc_S_Q", "auc_S_GS", "auc_classical_best",
         "oracle_auc", "delta", "gap_Q_MF", "shortfall_classical", "unreachable"]].median().reset_index()
    cur.to_csv(out / "SC_CURVES.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n{len(T)} of {len(M)} datasets scored")

    ql = T[(T.kind == "Q") & (T.readout == "local")]
    win = ql[ql.kappa.isin([2.0, 4.0]) & (ql.tau >= 3)]
    print("\n=== Q, local: median Delta (exact minus best opponent), rows p, cols kappa (tau 3-10) ===")
    print(ql[ql.tau >= 3].pivot_table(index="p", columns="kappa", values="delta", aggfunc="median").round(3).to_string())
    print("\n=== Q, local: median Delta, rows p, cols tau (kappa in {2,4}) ===")
    print(ql[ql.kappa.isin([2.0, 4.0])].pivot_table(index="p", columns="tau", values="delta", aggfunc="median").round(3).to_string())
    print("\n=== Q, local, window: median AUC by student, rows p ===")
    print(win.groupby("p")[["auc_S_Q", "auc_S_MF", "auc_S_J0", "auc_S_GS", "auc_classical_best", "oracle_auc",
                            "entanglement", "mf_agreement"]].median().round(3).to_string())
    mfl = T[(T.kind == "MF") & (T.readout == "local") & T.kappa.isin([2.0, 4.0]) & (T.tau >= 3)]
    print("\n=== MF teachers, local, window: classical shortfall and unreachable fraction by p ===")
    print(mfl.groupby("p").agg(shortfall=("shortfall_classical", "median"), unreachable=("unreachable", "mean"),
                               S_MF=("auc_S_MF", "median"), S_Q=("auc_S_Q", "median")).round(3).to_string())

    print("\n=== pre-registered checks ===")
    def _checks():
        rng = np.random.default_rng(0)
        x, yv = win.p.to_numpy(float), win.delta.to_numpy(float)
        slope = np.polyfit(x, yv, 1)[0]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) > 0:
                bs.append(np.polyfit(x[i], yv[i], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "shrinks" if hi < -0.005 else "grows" if lo > 0.005 else "flat/undetermined"
        print(f"S1 slope of Delta vs p: {slope:+.4f}/qubit, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}  (n={len(x)})")
        w3 = ql[ql.tau >= 3].assign(need=lambda t: t.gap_Q_MF >= 0.01)
        fr = w3.pivot_table(index="p", columns="kappa", values="need", aggfunc="mean")
        print("S2 fraction needing exact (S_Q-S_MF>=0.01), rows p, cols kappa:\n" + fr.round(2).to_string())
        pmax = max(fr.index)
        print(f"   kappa=1: p={min(fr.index)} {fr.loc[min(fr.index), 1.0]:.2f} -> p={pmax} {fr.loc[pmax, 1.0]:.2f}; "
              f"at p={pmax}: kappa=1 {fr.loc[pmax, 1.0]:.2f} vs kappa=1.5 {fr.loc[pmax, 1.5]:.2f}")
        qz = T[(T.kind == "Q") & (T.readout == "zlocal")]
        z = (qz.delta >= 0.01).groupby(qz.p).mean()
        print(f"S3 zlocal fraction Delta>=0.01 by p: {z.round(2).to_dict()}  -> {'PASS' if (z <= 0.10).all() else 'FAIL'}")
        rho = spearmanr(win.p, win.unreachable.astype(float)).correlation
        print(f"S4 window unreachable fraction by p: {win.groupby('p').unreachable.mean().round(2).to_dict()}; "
              f"Spearman(p, unreachable) = {rho:+.2f} -> {'PASS' if rho >= 0 else 'FAIL'}")
        rho6 = spearmanr(ql.mf_agreement, ql.delta, nan_policy="omit").correlation
        print(f"R  Spearman(mf_agreement, Delta), Q local: {rho6:+.2f}")
        print("R  per-sample entanglement vs |mean x| (Spearman), kappa=2/4, tau=5/10, by p:")
        for p in sorted(T.p.unique()):
            vals = []
            for k, t in ((2.0, 5.0), (2.0, 10.0), (4.0, 5.0), (4.0, 10.0)):
                try:
                    vals.append(f"{_balanced_rho(out, int(p), k, t):+.2f}")
                except FileNotFoundError:
                    vals.append("n/a")
            print(f"   p={int(p)}: " + "  ".join(vals))
    try:
        _checks()
    except Exception as e:
        print(f"   (checks incomplete: {type(e).__name__}: {e})")
    print(f"\nwrote {out / 'SC_DATASETS.csv'} and {out / 'SC_CURVES.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["make", "run", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="substring filter on dataset names (run phase)")
    ap.add_argument("--ps", nargs="+", type=int, help="make phase: which p to build (e.g. --ps 14 to add p=14)")
    a = ap.parse_args()
    if a.ps:
        DESIGN["ps"] = a.ps
    if a.phase == "make":
        do_make(a.out, a.workers)
    elif a.phase == "run":
        do_run(a.out, a.workers, a.only)
    else:
        do_score(a.out)


if __name__ == "__main__":
    main()
