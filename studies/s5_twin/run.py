#!/usr/bin/env python3
"""
S5: does a stronger classical twin (DTWA) copy the exact map in the window?
Plan and predictions: studies/s5_twin/PLAN.md (fixed before this code ran on real data).

Same rows, cells, readout weights and folds as S2, with the DTWA twin in place of mean-field.
The exact (Q) and mean-field (MF) banks are S2's (block 0 of the shared pool); nothing quantum
is simulated here.

Phases
  time     one row chunk at the largest p and tau; estimate of `banks` on --workers cores
  banks    DTWA banks for every (p, cell), M trajectories (M/4 kept for the noise check)
             -> runs/studies/s5/banks/p{p}_{cell}_M{M}.npz
           --flagged: only the cells the noise check flagged, at --M (e.g. 4096)
  measure  per (p, cell): U and E for DTWA and for mean-field on the same rows, DTWA's
           correlations against the exact ones
             -> runs/studies/s5/persample/*.npz, runs/studies/s5/S5_MEASURES.csv
  score    the noise check, exponents, checks D0-D4
             -> S5_NOISE_FLAGS.csv, S5_ALPHAS.csv, S5_CHECKS.csv

Usage (from the repo root):
  python studies/s5_twin/run.py time    --workers 31
  python studies/s5_twin/run.py banks   --workers 31 --M 1024
  python studies/s5_twin/run.py measure --workers 31
  python studies/s5_twin/run.py score
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
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

from aqfm_prescreen.aqfm import slices_for_p, strang_steps    # noqa: E402
from studies.common import bankstore as bs                    # noqa: E402
from studies.common import dtwa                               # noqa: E402
from studies.common.cells import cell_name, cells, coupling   # noqa: E402

_spec = importlib.util.spec_from_file_location("s2run_for_s5", ROOT / "studies" / "s2_meanfield" / "run.py")
S2 = importlib.util.module_from_spec(_spec)
sys.modules["s2run_for_s5"] = S2
_spec.loader.exec_module(S2)         # S2's U recipe: exact_scores, mf_design, oof_residuals
_spec1 = importlib.util.spec_from_file_location("s1run_for_s5", ROOT / "studies" / "s1_scaling" / "run.py")
S1 = importlib.util.module_from_spec(_spec1)
sys.modules["s1run_for_s5"] = S1
_spec1.loader.exec_module(S1)        # GPU pool and GPU check (Amendment 2)

DESIGN = {
    "version": "s5_1",
    "plan": "studies/s5_twin/PLAN.md",
    "ps": [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],   # S2's p; 14 and 16 only if `time` allows
    "test_p": 12,                                   # D2 is tested at p = 12
    "fit_p_min": 4,
    "M": 1024,
    "noise_tol": 0.02,                              # |U(M) - U(M/4)| above this -> rerun at 4M
    "n_boot": 1000,
    "chunk": 25,                                    # rows per piece (memory: chunk x M x p x 3)
}
CONVERGE = {                                        # Amendment 2 (PLAN Section 10): GPU, nested M
    "ps": [8, 12, 16],
    "Ms": [1024, 4096, 16384, 65536],
    "tol": 0.02,                                    # converged: |U(65536) - U(16384)| <= tol
    "chunk": 50,
}
BANKS = Path("runs/studies/banks/alltoall")          # S2's exact and mean-field banks
OUT = Path("runs/studies/s5")
MEASURES = ["U_DTWA", "U_MF", "E_DTWA", "E_MF"]


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def bank_file(p, cell, M):
    return OUT / "banks" / f"p{p}_{cell}_M{M}.npz"


def best_bank_file(p, cell):
    fs = sorted(OUT.glob(f"banks/p{p}_{cell}_M*.npz"), key=lambda f: int(f.stem.split("_M")[1]))
    return fs[-1] if fs else None


# ---------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------

def bank_job(args):
    p, c, tau, M = args
    cell = cell_name(c, tau)
    f = bank_file(p, cell, M)
    if f.exists():
        return "skip", f"p{p} {cell} M{M}"
    t0 = time.time()
    X = bs.pool_inputs(p)[:bs.BLOCK]
    rng = np.random.default_rng(_seed(f"s5_starts_p{p}_{cell}_M{M}"))
    B, Bs, M_used = dtwa.dtwa_bank(X, coupling(p, c, "Q"), tau, strang_steps(tau), M, rng=rng,
                                   chunk=DESIGN["chunk"], n_small=max(1, M // 4))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp.npz")
    np.savez_compressed(tmp, B=B, B_small=Bs, M_used=M_used)
    tmp.replace(f)
    return f"ok {time.time() - t0:.0f}s (M used {M_used})", f"p{p} {cell} M{M}"


def available_ps():
    """p with S2's exact and mean-field block 0 for every cell"""
    out = []
    for p in DESIGN["ps"]:
        if all(bs.block_path(BANKS, p, k, cl["c"], cl["tau"], 0).exists() for cl in cells() for k in ("Q", "MF")):
            out.append(p)
    return out


def _pool(fn, jobs, workers, label):
    t0, fails, res = time.time(), 0, []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
                res.append(r)
                if isinstance(r, tuple) and r[0] != "skip":
                    print(f"[{k}/{len(jobs)} {time.time() - t0:6.0f}s] {r[0]} {r[1]}", flush=True)
            except Exception:
                fails += 1
                print(f"[{k}/{len(jobs)}] FAILED {label} {futs[fu]}\n{traceback.format_exc()}", flush=True)
    return res, fails


def do_banks(workers, M, flagged=False, ps=None):
    ps = ps or available_ps()
    if not ps:
        raise SystemExit(f"STOP: S2's exact/mean-field banks not found under {BANKS}")
    if flagged:
        F = pd.read_csv(OUT / "S5_NOISE_FLAGS.csv")
        jobs = [(int(r.p), float(r.c), float(r.tau), M) for r in F.itertuples()]
    else:
        jobs = [(p, cl["c"], cl["tau"], M) for p in ps for cl in cells()]
    jobs.sort(key=lambda j: -(j[0] * j[0] * strang_steps(j[2])))      # biggest first
    print(f"{len(jobs)} DTWA bank jobs, M = {M}, p = {sorted({j[0] for j in jobs})}, {workers} workers", flush=True)
    _, fails = _pool(bank_job, jobs, workers, "bank")
    print(f"done, {fails} failed" + ("; rerun the same command (finished files are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def do_time(workers, M):
    p = max(available_ps() or [12])
    tau = max(cl["tau"] for cl in cells())
    X = bs.pool_inputs(p)[:5]
    t0 = time.time()
    dtwa.evolve(X, dtwa.initial_points(p, M), coupling(p, 4.0, "Q"), tau, 200)
    per_row_step = (time.time() - t0) / (5 * 200)
    total = 0.0
    for q in available_ps():
        for cl in cells():
            total += per_row_step * (q / p) * bs.BLOCK * strang_steps(cl["tau"])
    print(f"p = {p}, M = {M}: {per_row_step * 1e3:.2f} ms per row-step; all banks ~ {total / 3600:.1f} core-hours, "
          f"~ {total / workers / 60:.0f} min on {workers} workers (p = {available_ps()})")


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def _local(B, p):
    sl, _ = slices_for_p(p)
    return np.stack([B[:, sl[k]] for k in "XYZ"], -1).astype(float)          # (m, p, 3)


def connected_zz(B, p):
    """(m, q) connected ZZ correlations <Z_i Z_j> - <Z_i><Z_j>, pairs i < j"""
    sl, _ = slices_for_p(p)
    pairs = list(combinations(range(p), 2))
    ii = np.array([a for a, _ in pairs], int)
    jj = np.array([b for _, b in pairs], int)
    z = B[:, sl["Z"]].astype(float)
    return B[:, sl["ZZ"]].astype(float) - z[:, ii] * z[:, jj], ii, jj


def c_coll(B, p, J):
    """per row: J * mean_i | sum_{j != i} C^ZZ_ij | (S2's C_coll)"""
    C, ii, jj = connected_zz(B, p)
    m = len(B)
    Cm = np.zeros((m, p, p))
    Cm[:, ii, jj] = C
    Cm[:, jj, ii] = C
    return J * np.abs(Cm.sum(2)).mean(1)


def per_sample(BQ, BMF, BDT, BDT_small, p, c, cell):
    """per-row arrays behind every measure; U uses S2's exact teachers, design and folds"""
    rQ, rM, rD = _local(BQ, p), _local(BMF, p), _local(BDT, p)
    S = S2.exact_scores(BQ, p, cell)
    out = dict(
        e_dt=((rQ - rD) ** 2).sum(-1).mean(1), e_mf=((rQ - rM) ** 2).sum(-1).mean(1),
        r2_dt=S2.oof_residuals(S, S2.mf_design(BDT, p)),
        r2_dt_small=S2.oof_residuals(S, S2.mf_design(BDT_small, p)),
        r2_mf=S2.oof_residuals(S, S2.mf_design(BMF, p)),
        d2=(S - S.mean(0)) ** 2)
    if p >= 2:
        J = coupling(p, c, "Q")
        out["ccoll_q"] = c_coll(BQ, p, J)
        out["ccoll_dt"] = c_coll(BDT, p, J)
        cq, _, _ = connected_zz(BQ, p)
        cd, _, _ = connected_zz(BDT, p)
        out["czz_q"], out["czz_dt"] = cq.ravel(), cd.ravel()
    return out


def point_values(a):
    U = lambda r2: float(np.median(r2.sum(0) / a["d2"].sum(0)))
    u, us = U(a["r2_dt"]), U(a["r2_dt_small"])
    v = dict(U_DTWA=u, U_DTWA_small=us, U_DTWA_extrap=(4 * u - us) / 3, U_MF=U(a["r2_mf"]),
             E_DTWA=float(np.sqrt(a["e_dt"].mean())), E_MF=float(np.sqrt(a["e_mf"].mean())))
    if "ccoll_q" in a:
        v.update(C_coll_exact=float(a["ccoll_q"].mean()), C_coll_DTWA=float(a["ccoll_dt"].mean()),
                 corr_match=float(np.corrcoef(a["czz_q"], a["czz_dt"])[0, 1])
                 if a["czz_q"].std() > 0 and a["czz_dt"].std() > 0 else np.nan)
    return v


def measure_job(args):
    p, cl = args
    cell = cell_name(cl["c"], cl["tau"])
    fb = best_bank_file(p, cell)
    if fb is None:
        raise RuntimeError(f"no DTWA bank for p{p} {cell}")
    M = int(fb.stem.split("_M")[1])
    f = OUT / "persample" / f"p{p}_{cell}_M{M}.npz"
    if not f.exists():
        D = np.load(fb)
        BQ = bs.load_rows(BANKS, p, "Q", cl["c"], cl["tau"])
        BMF = bs.load_rows(BANKS, p, "MF", cl["c"], cl["tau"])
        a = per_sample(BQ, BMF, D["B"], D["B_small"], p, cl["c"], cell)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp.npz")
        np.savez_compressed(tmp, **a)
        tmp.replace(f)
    a = dict(np.load(f))
    M_used = int(np.load(fb)["M_used"])
    return dict(p=p, cell=cell, c=cl["c"], tau=cl["tau"], region=cl["region"], M=M, M_used=M_used,
                **point_values(a))


def do_measure(workers):
    ps = [p for p in available_ps() if all(best_bank_file(p, cell_name(cl["c"], cl["tau"])) for cl in cells())]
    if not ps:
        raise SystemExit("STOP: no complete DTWA banks; run `banks` first")
    jobs = [(p, cl) for p in ps for cl in cells()]
    print(f"measuring {len(jobs)} (p, cell) pairs for p = {ps}", flush=True)
    rows, fails = _pool(measure_job, jobs, workers, "measure")
    if fails:
        sys.exit(1)
    M = pd.DataFrame(rows).sort_values(["region", "c", "tau", "p"])
    M.to_csv(OUT / "S5_MEASURES.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["U_DTWA", "U_DTWA_small", "U_MF", "E_DTWA", "E_MF", "C_coll_exact", "C_coll_DTWA", "corr_match"]
    print("\nmedian over cells, by region and p:")
    print(M.groupby(["region", "p"])[cols].median().to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {OUT / 'S5_MEASURES.csv'}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def noise_flags(M):
    """cells whose U at M and at M/4 differ by more than noise_tol (enumerated cells never flag)"""
    M = M.assign(diff=(M.U_DTWA - M.U_DTWA_small).abs())
    return M[M["diff"] > DESIGN["noise_tol"]]


def bootstrap_U(ps, cell_list, n_boot, key="r2_dt"):
    """{cell: (n_boot, len(ps))} U recomputed on row resamples, the same resample for every cell"""
    out = {cl: np.empty((n_boot, len(ps))) for cl in cell_list}
    for j, p in enumerate(ps):
        A = {}
        for cl in cell_list:
            f = sorted(OUT.glob(f"persample/p{p}_{cl}_M*.npz"), key=lambda q: int(q.stem.split("_M")[1]))[-1]
            A[cl] = dict(np.load(f))
        m = len(next(iter(A.values()))["d2"])
        idx = np.random.default_rng(_seed(f"s5_boot_p{p}")).integers(0, m, (n_boot, m))
        for cl, a in A.items():
            out[cl][:, j] = np.median(a[key][idx].sum(1) / a["d2"][idx].sum(1), axis=1)
    return out


def checks(M, alpha_med, alpha_ci):
    tp = DESIGN["test_p"]
    win, sa = M[M.region == "window"], M[M.region == "control: short anneal"]
    rows = []
    ok0 = bool((sa.U_DTWA <= sa.U_MF + 1e-12).all() and (sa[sa.p >= 8].U_DTWA < 0.01).all())
    rows.append(("D0", "sanity: short-anneal control, U_DTWA <= U_MF at every p and < 0.01 at p >= 8",
                 "; ".join(f"p{int(r.p)}: {r.U_DTWA:.4f} vs {r.U_MF:.4f}" for r in sa.sort_values("p").itertuples()), ok0))
    cnt = win.assign(b=win.U_DTWA < win.U_MF).groupby("p").b.sum()
    rows.append(("D1", "DTWA beats mean-field (U_DTWA < U_MF) in >= 9 of 12 window cells at every p",
                 "; ".join(f"p{int(p)}: {int(c)}/12" for p, c in cnt.items()), bool((cnt >= 9).all())))
    w12 = win[win.p == tp]
    med = float(w12.U_DTWA.median()) if len(w12) else np.nan
    rows.append(("D2", f"the quantum part survives DTWA: median window U_DTWA at p = {tp} > 0.3",
                 f"{med:.3f} (mean-field: {float(w12.U_MF.median()):.3f})" if len(w12) else "p = 12 missing",
                 bool(med > 0.3)))
    rows.append(("D3", "reported: median exponent of U_DTWA over window cells (positive = DTWA catching up)",
                 f"{alpha_med:.3f} [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]", None))
    return pd.DataFrame(rows, columns=["check", "what", "value", "held"])


def do_score():
    M = pd.read_csv(OUT / "S5_MEASURES.csv")
    F = noise_flags(M)
    F[["p", "c", "tau", "cell", "M", "U_DTWA", "U_DTWA_small", "diff"]].to_csv(OUT / "S5_NOISE_FLAGS.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"noise check (|U at M - U at M/4| > {DESIGN['noise_tol']}): {len(F)} of {len(M)} (p, cell) flagged")
    if len(F):
        print(F[["p", "cell", "M", "U_DTWA", "U_DTWA_small", "diff"]].to_string(index=False))
        print("-> rerun those at 4M:  banks --flagged --M <4 x M>, then measure and score again")
    ps = sorted(p for p in M.p.unique() if p >= DESIGN["fit_p_min"])
    win_cells = sorted(M[M.region == "window"].cell.unique())
    A, rows = [], []
    boot = bootstrap_U(ps, win_cells, DESIGN["n_boot"])
    for cl in win_cells:
        v = M[M.cell == cl].set_index("p").loc[ps].U_DTWA.values
        a, _ = S2.fit_alpha(ps, v)
        ab = np.array([S2.fit_alpha(ps, boot[cl][b])[0] for b in range(DESIGN["n_boot"])])
        rows.append(dict(cell=cl, alpha=a, lo=np.percentile(ab, 2.5), hi=np.percentile(ab, 97.5)))
        A.append(ab)
    Al = pd.DataFrame(rows)
    Al.to_csv(OUT / "S5_ALPHAS.csv", index=False)
    med_boot = np.median(np.stack(A), axis=0)
    alpha_med = float(Al.alpha.median())
    K = checks(M, alpha_med, (float(np.percentile(med_boot, 2.5)), float(np.percentile(med_boot, 97.5))))
    K.to_csv(OUT / "S5_CHECKS.csv", index=False)
    print("\nU (share of the exact labels the twin can't explain), median over cells:")
    t = M.groupby(["region", "p"])[["U_DTWA", "U_MF"]].median().unstack("region")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    if "U_DTWA_extrap" in M:
        print("\nreported, not a test (PLAN Section 9): U_DTWA extrapolated to infinitely many trajectories,\n"
              "(4 U(M) - U(M/4)) / 3, assuming the sampling bias shrinks like 1/M; window, median over cells:")
        w = M[M.region == "window"].groupby("p")[["U_DTWA_small", "U_DTWA", "U_DTWA_extrap", "U_MF"]].median()
        print(w.rename(columns={"U_DTWA_small": "U(M/4)", "U_DTWA": "U(M)", "U_DTWA_extrap": "U(M->inf)"})
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print("M per cell (largest available):", M.groupby("M").size().to_dict())
    print("\nwindow U_DTWA by anneal time (median over c):")
    print(M[M.region == "window"].groupby(["tau", "p"]).U_DTWA.median().unstack("p").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nwindow: DTWA's correlations vs exact (median over cells): C_coll exact, C_coll DTWA, correlation of ZZ values")
    print(M[M.region == "window"].groupby("p")[["C_coll_exact", "C_coll_DTWA", "corr_match"]].median()
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nexponent of U_DTWA per window cell (fit on p >= 4):")
    print(Al.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== S5 checks ===")
    print(K.to_string(index=False))
    if len(F):
        print("\nNOTE: the noise check flagged cells; the checks above are provisional until they are rerun at 4M.")
    print(f"\nwrote {OUT / 'S5_CHECKS.csv'}, {OUT / 'S5_ALPHAS.csv'}, {OUT / 'S5_NOISE_FLAGS.csv'}")


# ---------------------------------------------------------------------------
# Amendment 2: convergence in M on the GPUs (PLAN Section 10)
# ---------------------------------------------------------------------------

def converge_job(args):
    """one (p, cell): DTWA with nested M = 1,024 ... 65,536 from the same trajectories, on a GPU;
    U and E at every M on S2's rows, readouts and folds; signal-to-noise of the features"""
    p, c, tau, smoke = args
    cell = cell_name(c, tau)
    out = (Path("runs/studies/s5_smoke") if smoke else OUT) / "converge"
    f = out / f"p{p}_{cell}.csv"
    label = f"converge p{p} {cell}"
    if f.exists():
        return "skip", label
    import torch
    from studies.common import torchdtwa as td
    torch.set_num_threads(1)
    dev = os.environ.get(bs.TORCH_ENV, "cpu")
    Ms = [256, 1024] if smoke else CONVERGE["Ms"]
    t0 = time.time()
    X = bs.pool_inputs(p)[:bs.BLOCK]
    rng = np.random.default_rng(_seed(f"s5c_starts_p{p}_{cell}"))
    R = td.nested_banks(X, coupling(p, c, "Q"), tau, strang_steps(tau), Ms, rng, dev, chunk=CONVERGE["chunk"])
    BQ = bs.load_rows(BANKS, p, "Q", c, tau)
    BMF = bs.load_rows(BANKS, p, "MF", c, tau)
    S = S2.exact_scores(BQ, p, cell)
    d2 = (S - S.mean(0)) ** 2
    rQ = _local(BQ, p)
    rows = []
    for M in sorted(k for k in R if isinstance(k, int)):          # the nested trajectory counts
        B = R[M]
        r2 = S2.oof_residuals(S, S2.mf_design(B, p))
        loc = B[:, :3 * p]
        sig = loc.var(0)                                             # spread of each feature over rows
        noise = R["loc_var"].mean(0) / M                             # sampling variance of each average
        rows.append(dict(p=p, cell=cell, c=c, tau=tau, M=M, M_used=R["M_used"],
                         U_DTWA=float(np.median(r2.sum(0) / d2.sum(0))),
                         E_DTWA=float(np.sqrt(((rQ - _local(B, p)) ** 2).sum(-1).mean())),
                         snr_median=float(np.median(sig / np.maximum(noise, 1e-30))),
                         snr_min=float(np.min(sig / np.maximum(noise, 1e-30)))))
    U_MF = float(np.median(S2.oof_residuals(S, S2.mf_design(BMF, p)).sum(0) / d2.sum(0)))
    for r in rows:
        r["U_MF"] = U_MF
    S1._write(f, rows)
    return f"ok {time.time() - t0:.0f}s (M used {R['M_used']})", label


def do_converge(gpus, smoke):
    if gpus < 1 or gpus > 31:
        raise SystemExit("STOP: --gpus must be 1-31")
    S1.require_gpus(gpus)
    ps = [4] if smoke else CONVERGE["ps"]
    cl = cells()[:2] if smoke else cells()
    jobs = [(p, x["c"], x["tau"], smoke) for p in ps for x in cl]
    jobs.sort(key=lambda j: -(j[0] * strang_steps(j[2])))
    print(f"{len(jobs)} convergence jobs on {gpus} GPU(s), p = {ps}, M = {[256, 1024] if smoke else CONVERGE['Ms']}",
          flush=True)
    fails = S1._run_gpu_pool(jobs, converge_job, gpus)
    print(f"done, {fails} failed" + ("; rerun the same command (finished files are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def do_converge_score():
    files = sorted((OUT / "converge").glob("*.csv"))
    if not files:
        raise SystemExit("STOP: no convergence files")
    C = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    reg = {cell_name(x["c"], x["tau"]): x["region"] for x in cells()}
    C["region"] = C.cell.map(reg)
    C.to_csv(OUT / "S5_CONVERGE.csv", index=False)
    pd.set_option("display.width", 220)
    Mmax = C.M.max()
    W = C.pivot_table(index=["region", "c", "tau", "p"], columns="M", values="U_DTWA")
    print("U_DTWA by number of trajectories (same trajectories, nested), every cell:")
    print(W.to_string(float_format=lambda v: f"{v:.3f}"))
    prev = sorted(C.M.unique())[-2]
    conv = (W[Mmax] - W[prev]).abs()
    print(f"\nconverged (|U({Mmax}) - U({prev})| <= {CONVERGE['tol']}): {int((conv <= CONVERGE['tol']).sum())} "
          f"of {len(conv)} (p, cell)")
    if (conv > CONVERGE["tol"]).any():
        print(conv[conv > CONVERGE["tol"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsignal-to-noise of the DTWA features (median over features; spread over rows / sampling variance):")
    print(C.pivot_table(index=["region", "c", "tau", "p"], columns="M", values="snr_median")
          .to_string(float_format=lambda v: f"{v:.0f}"))
    w = C[(C.region == "window") & (C.M == Mmax)]
    tp = DESIGN["test_p"]
    med = float(w[w.p == tp].U_DTWA.median())
    print(f"\nwindow, M = {Mmax}: median U_DTWA by p (mean-field in brackets)")
    t = w.groupby("p").agg(U_DTWA=("U_DTWA", "median"), U_MF=("U_MF", "median"))
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nexploratory (seen in the first scoring, not a pre-registered split): window by coupling c, M = "
          f"{Mmax}, median U_DTWA")
    print(w.groupby(["c", "p"]).U_DTWA.median().unstack("p").to_string(float_format=lambda v: f"{v:.3f}"))
    cw = conv.reset_index()
    n_unconv = int(((cw.region == "window") & (cw.p == tp) & (cw[0] > CONVERGE["tol"])).sum())
    settled = n_unconv <= 3
    K = pd.DataFrame([("D2 (Amendment 2)", f"median window U_DTWA at p = {tp}, M = {Mmax} > 0.3 "
                       f"(unsettled if > 3 of 12 window cells at p = {tp} are not converged)",
                       f"{med:.3f}; {n_unconv}/12 window cells not converged" + ("" if settled else " -> UNSETTLED"),
                       (med > 0.3) if settled else None),
                      ("convergence", f"cells with |U({Mmax}) - U({prev})| <= {CONVERGE['tol']}",
                       f"{int((conv <= CONVERGE['tol']).sum())}/{len(conv)}", None)],
                     columns=["check", "what", "value", "held"])
    K.to_csv(OUT / "S5_CONVERGE_CHECKS.csv", index=False)
    print("\n=== S5 Amendment 2 ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S5_CONVERGE.csv'}, {OUT / 'S5_CONVERGE_CHECKS.csv'}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["time", "banks", "measure", "score", "converge", "converge_score"])
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--M", type=int, default=DESIGN["M"])
    ap.add_argument("--ps", default=None, help="comma-separated p (default: every p with S2 banks)")
    ap.add_argument("--flagged", action="store_true")
    a = ap.parse_args()
    if a.workers > 31:
        raise SystemExit("STOP: at most 31 workers (32-core limit)")
    OUT.mkdir(parents=True, exist_ok=True)
    ps = [int(x) for x in a.ps.split(",")] if a.ps else None
    if a.phase == "time":
        do_time(a.workers, a.M)
    elif a.phase == "banks":
        do_banks(a.workers, a.M, a.flagged, ps)
    elif a.phase == "measure":
        do_measure(a.workers)
    elif a.phase == "score":
        do_score()
    elif a.phase == "converge":
        do_converge(a.gpus, a.smoke)
    else:
        do_converge_score()


if __name__ == "__main__":
    main()
