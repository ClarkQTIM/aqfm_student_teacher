#!/usr/bin/env python3
"""
S1: how does the classical cost of matching the map grow with the number of qubits p?
Plan and predictions: studies/s1_scaling/PLAN.md (fixed before any run).

This file covers the data, PART A (how complex the label function is, no learners) and PART B
(how well classical learners do, by size and by number of labelled rows; PLAN.md, Amendment 1).

Phases
  banks    simulate the banks every teacher needs, pool rows 0-39,999 (100 blocks of 400):
             exact (Q) on the GPUs (--gpus N, one worker per GPU, PyTorch),
             mean-field (MF) and J = 0 on CPU (--cpu-workers M), at the same time.
             Total processes = N + M <= 31.
             -> runs/studies/banks/alltoall/p{p}/... (shared with S2; existing blocks skipped)
  teachers the noise-free scores and the 0/1 labels of every teacher, all 40,000 rows
             -> runs/studies/s1/teachers/p{p}.npz, S1_TEACHERS.csv
  partA    for every teacher: test R^2 of a least-squares polynomial fit (Hermite basis) of its
           noise-free score, degree d = 1 ... d_max(p), trained on rows 0-29,999, tested on
           rows 30,000-39,999        -> runs/studies/s1/S1_PARTA.csv
  scoreA   the checks A1-A4          -> runs/studies/s1/S1_PARTA_CHECKS.csv
  partB_gpu    on the GPUs (--gpus N): for every teacher at p = 4-12 and n = 300 ... 30,000
               labelled rows (80% train, 20% validation; test = rows 30,000-39,999):
                 the open-book reference (logistic regression on the teacher's own map features),
                 fixed bases + logistic regression (Hermite polynomials, Fourier, random Fourier),
                 neural networks (width x depth grid)
                                     -> runs/studies/s1/partB/{ref,basis,mlp}_*.csv
  partB_trees  on CPU (--workers M): gradient-boosted trees (leaves x rounds grid)
                                     -> runs/studies/s1/partB/gbt_*.csv
               (the two can run at the same time: N + M <= 31)
  scoreB   each learner family picks its size on validation rows; the checks B0-B4
                                     -> runs/studies/s1/S1_PARTB.csv, S1_PARTB_CHECKS.csv
  --smoke  (partB_gpu / partB_trees) a tiny version at p = 4 written to runs/studies/s1_smoke/,
           to check the code end to end on the server; it prints timings only, no AUCs

Usage (from the repo root; see the chat for the copy-paste block):
  python studies/s1_scaling/run.py banks    --ps 2,3,4,5,6,8,10,12 --gpus 2 --cpu-workers 29
  python studies/s1_scaling/run.py teachers --workers 31
  python studies/s1_scaling/run.py partA    --workers 8
  python studies/s1_scaling/run.py scoreA
  python studies/s1_scaling/run.py partB_gpu   --gpus 2
  python studies/s1_scaling/run.py partB_trees --workers 29
  python studies/s1_scaling/run.py scoreB
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: workers x 1 <= 32 cores

import argparse
import hashlib
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement
from math import comb, factorial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import family_matrix, strang_steps   # noqa: E402
from studies.common import bankstore as bs                     # noqa: E402
from studies.common.cells import cells, coupling                # noqa: E402

DESIGN = {
    "version": "s1_1",
    "plan": "studies/s1_scaling/PLAN.md",
    "ps": [2, 3, 4, 5, 6, 8, 10, 12],
    "n_blocks": 100,               # 100 x 400 = 40,000 pool rows
    "train_rows": 30_000,          # rows 0 .. 29,999: training pool (Part B draws nested n from it)
    "test_rows": 10_000,           # rows 30,000 .. 39,999: the fixed test set
    "seeds": 3,                    # readout weight vectors per teacher kind and cell
    "temp": 0.25,                  # label temperature, as in the tau sweep
    "max_terms": 3_000,            # Part A: at most 3,000 polynomial terms (see PLAN, Section 5)
    "max_degree": 12,
    "fit_ps": [4, 5, 6, 8, 10, 12],  # slopes are fitted on p >= 4
    "primary_degree": 3,
    "secondary_degree": 4,
    "n_boot": 2000,
}
# Part B (PLAN.md, Amendment 1).  Fixed before Part B's code ran on any real data.
PARTB = {
    "ps": [4, 5, 6, 8, 10, 12],                  # = Part A's fit_ps
    "n_grid": [300, 1_000, 3_000, 10_000, 30_000],   # labelled rows: first n rows of the training pool
    "val_frac": 0.2,                             # last 20% of the n rows: validation (size, penalty, stopping)
    "lams": [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], # L2 penalties for every logistic regression
    "fourier_mults": [1, 4, 16, 64, 256],        # Fourier and random-Fourier sizes D = p x mult
    "mlp_widths": [16, 64, 256, 1024],
    "mlp_depths": [2, 3, 4],                     # hidden layers
    "mlp": dict(lr=1e-3, batch=512, eval_every=50, patience=40, max_steps=20_000),
    "gbt_leaves": [8, 31, 127],
    "gbt_rounds": [100, 300, 1_000, 3_000],
    "gbt_lr": 0.1,
    "gbt_chunk": 33,                             # teachers per tree job
    "gets_there": 0.01,                          # AUC within 0.01 of the open-book reference
    "test_n": 30_000,                            # the n at which B0-B3 are tested
    "test_p": 8,                                 # B2 is tested at p = 8
}
SMOKE = {
    "ps": [4], "n_grid": [300, 1_000], "fourier_mults": [1, 4], "mlp_widths": [16, 64],
    "mlp_depths": [2], "mlp": dict(lr=1e-3, batch=512, eval_every=25, patience=4, max_steps=300),
    "gbt_leaves": [8], "gbt_rounds": [10, 30], "gbt_chunk": 50, "max_degree_B": 2,
}
BANKS = Path("runs/studies/banks/alltoall")
OUT = Path("runs/studies/s1")


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def n_rows():
    return DESIGN["n_blocks"] * bs.BLOCK


# ---------------------------------------------------------------------------
# teachers: which banks they need
# ---------------------------------------------------------------------------

def teacher_specs(p):
    """every teacher at p: exact and mean-field at each of the 14 cells, J = 0 at each anneal
    time the cells use; 3 readout seeds each.  Region of a J = 0 teacher = 'window' if its tau
    is a window tau (3, 5, 10, 20), else the control's region."""
    out = []
    for cl in cells():
        for kind in ("Q", "MF"):
            for s in range(DESIGN["seeds"]):
                out.append(dict(p=p, kind=kind, c=cl["c"], tau=cl["tau"], seed=s, region=cl["region"]))
    win_taus = sorted({cl["tau"] for cl in cells() if cl["region"] == "window"})
    other = {cl["tau"]: cl["region"] for cl in cells() if cl["region"] != "window"}
    for t in sorted({cl["tau"] for cl in cells()}):
        for s in range(DESIGN["seeds"]):
            out.append(dict(p=p, kind="J0", c=0.0, tau=t, seed=s,
                            region="window" if t in win_taus else other[t]))
    for d in out:
        d["name"] = f"s1_{d['kind']}_p{p}_c{d['c']:g}_t{d['tau']:g}_s{d['seed']}"
    return out


def bank_keys(p):
    """unique (kind, c, tau) banks the teachers at p need"""
    return list(dict.fromkeys((t["kind"], t["c"], t["tau"]) for t in teacher_specs(p)))


# ---------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------

def _gpu_init(q):
    """runs once in each GPU worker: claim one GPU index"""
    os.environ[bs.TORCH_ENV] = f"cuda:{q.get()}"


GPU_BLOCKS_PER_JOB = 10          # 4,000 rows per GPU job: fewer, bigger kernels


def gpu_job(args):
    """one GPU job: simulate several missing exact blocks of one bank in a single batch
    (studies/common/torchsim.py) and save each block exactly as the CPU path would"""
    root, p, c, tau, blocks = args
    todo = [b for b in blocks if not bs.block_path(root, p, "Q", c, tau, b).exists()]
    label = f"p{p} Q c{c:g} t{tau:g} blocks {blocks[0]}-{blocks[-1]}"
    if not todo:
        return "skip", label
    from studies.common.torchsim import compute_bank_torch      # only GPU workers import torch
    t0 = time.time()
    Z = bs.pool_inputs(p)
    rows = np.concatenate([np.arange(b * bs.BLOCK, (b + 1) * bs.BLOCK) for b in todo])
    B = compute_bank_torch(Z[rows], coupling(p, c, "Q"), tau, device=os.environ[bs.TORCH_ENV])
    for i, b in enumerate(todo):
        bs._save(bs.block_path(root, p, "Q", c, tau, b), B[i * bs.BLOCK:(i + 1) * bs.BLOCK])
    return f"ok {time.time() - t0:.0f}s", label


def _gpu_jobs(ps):
    jobs = []
    for p in ps:
        for kind, c, t in bank_keys(p):
            if kind != "Q":
                continue
            missing = [b for b in range(DESIGN["n_blocks"]) if not bs.block_path(BANKS, p, "Q", c, t, b).exists()]
            for i in range(0, len(missing), GPU_BLOCKS_PER_JOB):
                jobs.append((BANKS, p, c, t, missing[i:i + GPU_BLOCKS_PER_JOB]))
    jobs.sort(key=lambda j: -((2.0 ** j[1]) * strang_steps(j[3])))          # slowest first
    return jobs


def _piece_jobs(ps, kinds):
    jobs = []
    for p in ps:
        for kind, c, t in bank_keys(p):
            if kind not in kinds:
                continue
            for b in range(DESIGN["n_blocks"]):
                if bs.block_path(BANKS, p, kind, c, t, b).exists():
                    continue
                for s, e in bs.piece_specs(BANKS, p, kind, c, t, b):
                    jobs.append((BANKS, p, kind, c, t, b, s, e))
    jobs.sort(key=lambda j: -((2.0 ** j[1]) * strang_steps(j[4]) * (j[7] - j[6])))   # slowest first
    return jobs


def do_banks(ps, gpus, cpu_workers):
    if gpus + cpu_workers > 31:
        raise SystemExit("STOP: --gpus + --cpu-workers must be <= 31 (32-core limit)")
    q_jobs = _gpu_jobs(ps) if gpus else []
    c_jobs = _piece_jobs(ps, {"MF", "J0"} if gpus else {"Q", "MF", "J0"})
    print(f"{len(q_jobs)} exact jobs ({GPU_BLOCKS_PER_JOB} blocks each) on {gpus} GPU(s); "
          f"{len(c_jobs)} pieces on {cpu_workers} CPU workers", flush=True)
    t0, fails, done = time.time(), 0, 0
    executors, futs = [], {}
    # CPU pool FIRST: its workers are forked, and forking is only safe before any thread
    # exists in this process.  With the default fork start method, ProcessPoolExecutor
    # forks all its workers on the first submit, before starting its own manager thread.
    # The GPU pool uses fresh (spawned) processes, so starting it afterwards is safe.
    if c_jobs:
        c = ProcessPoolExecutor(cpu_workers, mp_context=mp.get_context("fork"))
        executors.append(c)
        futs.update({c.submit(bs.compute_piece, j): j for j in c_jobs})
    if gpus and q_jobs:
        ctx = mp.get_context("spawn")           # CUDA needs fresh processes, not forks
        idq = ctx.Queue()
        for i in range(gpus):
            idq.put(i)
        g = ProcessPoolExecutor(gpus, mp_context=ctx, initializer=_gpu_init, initargs=(idq,))
        executors.append(g)
        futs.update({g.submit(gpu_job, j): ("gpu",) + j[1:] for j in q_jobs})
    total = len(futs)
    for fu in as_completed(futs):
        done += 1
        try:
            st, nm = fu.result()
            if st != "skip" and (done % 20 == 0 or futs[fu][0] == "gpu" or done == total):
                print(f"[{done}/{total} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
        except Exception:
            fails += 1
            print(f"[{done}/{total}] FAILED {futs[fu][1:]}\n{traceback.format_exc()}", flush=True)
    for e in executors:
        e.shutdown()
    print(f"done, {fails} failed" + ("; rerun to retry (finished pieces are skipped)" if fails else ""))
    missing = 0
    for p in ps:
        for kind, cc, t in bank_keys(p):
            for b in range(DESIGN["n_blocks"]):
                missing += not bs.merge_block(BANKS, p, kind, cc, t, b)
    if fails or missing:
        print(f"STOP: {missing} bank blocks incomplete (rerun the same command to finish)")
        sys.exit(1)
    print(f"all banks complete for p = {ps}")


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------

def _standardize(F):
    sd = F.std(0)
    keep = sd > 1e-9
    return (F[:, keep] - F[:, keep].mean(0)) / sd[keep]


def teachers_job(p):
    """noise-free scores S (rows x teachers) and labels Y for every teacher at p.
    The tau sweep's recipe on the full readout (<X>, <Y>, <Z> per qubit): standardize the
    features over all 40,000 rows, random weights w ~ N(0, I), score = standardized(F w),
    y ~ Bernoulli(sigmoid(score / T)).  Weights and coin flips are seeded by the teacher's name."""
    f = OUT / "teachers" / f"p{p}.npz"
    specs = teacher_specs(p)
    if not f.exists():
        S = np.empty((n_rows(), len(specs)), np.float32)
        Y = np.empty((n_rows(), len(specs)), np.int8)
        cache = {}
        for k, t in enumerate(specs):
            key = (t["kind"], t["c"], t["tau"])
            if key not in cache:
                B = bs.load_rows(BANKS, p, *key, DESIGN["n_blocks"])
                cache = {key: _standardize(family_matrix(B, "XYZ", p).astype(float))}
            F = cache[key]
            rng = np.random.default_rng(_seed(t["name"]))
            s = F @ rng.standard_normal(F.shape[1])
            s = (s - s.mean()) / s.std()
            S[:, k] = s
            coin = np.random.default_rng(_seed(t["name"] + "_coin")).random(len(s))
            Y[:, k] = (coin < 1.0 / (1.0 + np.exp(-s / DESIGN["temp"]))).astype(np.int8)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp.npz")
        np.savez_compressed(tmp, S=S, Y=Y, names=np.array([t["name"] for t in specs]))
        tmp.replace(f)
    return p, "ok"


def do_teachers(ps, workers):
    with ProcessPoolExecutor(min(workers, len(ps))) as ex:
        for fu in as_completed([ex.submit(teachers_job, p) for p in ps]):
            print("teachers", *fu.result(), flush=True)
    M = pd.DataFrame([t for p in ps for t in teacher_specs(p)])
    for p in ps:
        Y = np.load(OUT / "teachers" / f"p{p}.npz")["Y"]
        M.loc[M.p == p, "pos_rate"] = Y.mean(0)
    M.to_csv(OUT / "S1_TEACHERS.csv", index=False)
    print(f"wrote {OUT / 'S1_TEACHERS.csv'} ({len(M)} teachers); positive rate "
          f"{M.pos_rate.min():.2f}-{M.pos_rate.max():.2f}")


# ---------------------------------------------------------------------------
# Part A: polynomial degree profile of the noise-free score
# ---------------------------------------------------------------------------

def d_max(p):
    """largest degree d <= max_degree whose number of non-constant terms C(p+d, d) - 1 <= max_terms"""
    d = 0
    while d < DESIGN["max_degree"] and comb(p + d + 1, d + 1) - 1 <= DESIGN["max_terms"]:
        d += 1
    return d


def hermite_tables(Z, d):
    """H[j][:, k] = He_k(z_j) / sqrt(k!), the probabilists' Hermite polynomials, normalized so
    they are orthonormal for z ~ N(0, 1).  Recurrence He_{k+1} = z He_k - k He_{k-1}."""
    n, p = Z.shape
    H = np.empty((p, n, d + 1))
    for j in range(p):
        z = Z[:, j]
        h = [np.ones(n), z.copy()]
        for k in range(1, d):
            h.append(z * h[k] - k * h[k - 1])
        h = h[:d + 1]
        for k in range(d + 1):
            H[j, :, k] = h[k] / np.sqrt(float(factorial(k)))
    return H


def hermite_design(Z, d):
    """all products of Hermite polynomials of total degree <= d, ORDERED BY DEGREE, so the
    first C(p+t, t) columns are exactly the degree <= t design for every t <= d"""
    n, p = Z.shape
    H = hermite_tables(Z, d)
    cols = [np.ones(n)]
    for t in range(1, d + 1):
        for combo in combinations_with_replacement(range(p), t):
            col = np.ones(n)
            for j, k in Counter(combo).items():
                col = col * H[j, :, k]
            cols.append(col)
    X = np.stack(cols, 1)
    assert X.shape[1] == comb(p + d, d)
    return X


def degree_profile(X_tr, S_tr, X_te, S_te, p, dmax):
    """test R^2 of least squares on the degree <= d design, d = 1 .. dmax, for every column of S.
    One Gram matrix at dmax; each lower degree uses its leading block (columns are degree-ordered)."""
    G = X_tr.T @ X_tr
    B = X_tr.T @ S_tr
    sst = ((S_te - S_te.mean(0)) ** 2).sum(0)
    out = {}
    for d in range(1, dmax + 1):
        m = comb(p + d, d)
        Gd = G[:m, :m] + 1e-10 * np.trace(G[:m, :m]) / m * np.eye(m)     # tiny ridge, for stability
        coef = np.linalg.solve(Gd, B[:m])
        res = S_te - X_te[:, :m] @ coef
        out[d] = 1.0 - (res ** 2).sum(0) / sst
    return out


def partA_job(p):
    f = OUT / "partA" / f"p{p}.csv"
    if not f.exists():
        T = np.load(OUT / "teachers" / f"p{p}.npz")
        S = T["S"].astype(float)
        names = T["names"]
        Z = bs.pool_inputs(p)[:n_rows()]
        ntr = DESIGN["train_rows"]
        dm = d_max(p)
        X = hermite_design(Z, dm)
        prof = degree_profile(X[:ntr], S[:ntr], X[ntr:], S[ntr:], p, dm)
        rows = [dict(name=nm, d=d, D=comb(p + d, d) - 1, R2=float(prof[d][k]))
                for d in prof for k, nm in enumerate(names)]
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f, index=False)
    return p, f"ok (d_max = {d_max(p)})"


def do_partA(ps, workers):
    with ProcessPoolExecutor(min(workers, len(ps))) as ex:
        for fu in as_completed([ex.submit(partA_job, p) for p in ps]):
            print("partA", *fu.result(), flush=True)
    M = pd.read_csv(OUT / "S1_TEACHERS.csv")
    A = pd.concat([pd.read_csv(OUT / "partA" / f"p{p}.csv") for p in ps]).merge(M, on="name")
    A["V"] = 1.0 - A.R2
    A.to_csv(OUT / "S1_PARTA.csv", index=False)
    print(f"wrote {OUT / 'S1_PARTA.csv'}")


# ---------------------------------------------------------------------------
# Part A checks
# ---------------------------------------------------------------------------

def unit_slopes(A, d):
    """slope of V_d (variance beyond degree d) against p, one per teacher unit (kind, c, tau, seed),
    fitted on p in fit_ps"""
    sub = A[(A.d == d) & A.p.isin(DESIGN["fit_ps"])]
    rows = []
    for key, g in sub.groupby(["kind", "c", "tau", "seed", "region"]):
        g = g.sort_values("p")
        if len(g) < 3:
            continue
        rows.append(dict(zip(["kind", "c", "tau", "seed", "region"], key),
                         slope=float(np.polyfit(g.p, g.V, 1)[0])))
    return pd.DataFrame(rows, columns=["kind", "c", "tau", "seed", "region", "slope"])


def _boot_median(v, rng, n):
    v = np.asarray(v)
    return np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)


def _ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def checks_partA(A, d):
    U = unit_slopes(A, d)
    rng = np.random.default_rng(_seed(f"s1_scoreA_d{d}"))
    n = DESIGN["n_boot"]
    grp = {
        "exact, window": U[(U.kind == "Q") & (U.region == "window")].slope,
        "exact, controls": U[(U.kind == "Q") & (U.region != "window")].slope,
        "mean-field, window": U[(U.kind == "MF") & (U.region == "window")].slope,
        "J=0, window taus": U[(U.kind == "J0") & (U.region == "window")].slope,
    }
    bm = {k: _boot_median(v, rng, n) for k, v in grp.items()}
    med = {k: float(np.median(v)) for k, v in grp.items()}
    rows = []
    lo, hi = _ci(bm["exact, window"])
    rows.append(("A1", f"growth: median slope of V_{d} (exact, window) > 0",
                 f"{med['exact, window']:.4f} [{lo:.4f}, {hi:.4f}] per qubit", lo > 0))
    for other in ("exact, controls", "J=0, window taus"):
        diff = bm["exact, window"] - bm[other]
        lo, hi = _ci(diff)
        rows.append(("A2", f"beyond input dimension: exact window minus {other} > 0",
                     f"{med['exact, window'] - med[other]:.4f} [{lo:.4f}, {hi:.4f}]", lo > 0))
    diff = bm["exact, window"] - bm["mean-field, window"]
    lo, hi = _ci(diff)
    outcome = "contains 0 (as expected)" if lo <= 0 <= hi else ("above 0: outcome (Q)" if lo > 0 else "below 0")
    rows.append(("A3", "not quantum-specific: exact minus mean-field (window), interval contains 0",
                 f"{med['exact, window'] - med['mean-field, window']:.4f} [{lo:.4f}, {hi:.4f}]: {outcome}",
                 lo <= 0 <= hi))
    K = pd.DataFrame(rows, columns=["check", "what", "value", "held"])
    K.insert(0, "degree", d)
    return K, med


def censored_degree(A, level):
    """smallest d with R^2 >= level per teacher; '> d_max' if never reached"""
    out = []
    for nm, g in A.groupby("name"):
        ok = g[g.R2 >= level].d
        out.append(dict(name=nm, d=int(ok.min()) if len(ok) else np.inf))
    return pd.DataFrame(out)


def do_scoreA():
    A = pd.read_csv(OUT / "S1_PARTA.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    lab = lambda r: f"{r.kind}, {'window' if r.region == 'window' else 'controls'}"
    A["group"] = A.apply(lab, axis=1)
    print("\nV_d = share of the noise-free score NOT captured by polynomials of degree <= d "
          "(test set; median over teachers)")
    for d in (1, 2, DESIGN["primary_degree"], DESIGN["secondary_degree"]):
        t = A[A.d == d].groupby(["group", "p"]).V.median().unstack("p")
        print(f"\ndegree {d}:")
        print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    for lvl in (0.90, 0.99):
        C = censored_degree(A, lvl).merge(A[["name", "group", "p"]].drop_duplicates(), on="name")
        t = C.groupby(["group", "p"]).d.median().unstack("p")
        print(f"\ndegree needed for R^2 >= {lvl} (median; inf = beyond d_max(p) = "
              f"{ {int(p): d_max(int(p)) for p in sorted(A.p.unique())} }):")
        print(t.to_string(float_format=lambda v: f"{v:.0f}"))
    Ks = []
    for d in (DESIGN["primary_degree"], DESIGN["secondary_degree"]):
        K, med = checks_partA(A, d)
        Ks.append(K)
        print(f"\nmedian slope of V_{d} per qubit (p = {DESIGN['fit_ps']}): "
              + "; ".join(f"{k} {v:.4f}" for k, v in med.items()))
    K = pd.concat(Ks)
    K.to_csv(OUT / "S1_PARTA_CHECKS.csv", index=False)
    print(f"\n=== Part A checks (degree {DESIGN['primary_degree']} is the test; "
          f"degree {DESIGN['secondary_degree']} is reported) ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S1_PARTA_CHECKS.csv'}")


# ---------------------------------------------------------------------------
# Part B: classical learners by size and by number of labelled rows
# ---------------------------------------------------------------------------

def partB_cfg(smoke=False):
    cfg = dict(PARTB)
    if smoke:
        cfg.update(SMOKE)
    return cfg


def partB_dir(smoke=False):
    return Path("runs/studies/s1_smoke/partB") if smoke else OUT / "partB"


def splits(n, val_frac):
    """row indices into the pool: train = first (1 - val_frac) n rows, validation = the rest of
    the first n rows, test = the fixed rows 30,000-39,999.  Nested in n."""
    k = int(round(n * (1 - val_frac)))
    ntr = DESIGN["train_rows"]
    return np.arange(k), np.arange(k, n), np.arange(ntr, ntr + DESIGN["test_rows"])


def load_teachers(p):
    """labels Y (40,000 x M) and names of every teacher at p (written by the `teachers` phase)"""
    T = np.load(OUT / "teachers" / f"p{p}.npz")
    return T["Y"].astype(np.float32), [str(x) for x in T["names"]]


def basis_sizes(p, cfg):
    """the fixed bases at p: (family, size label, D)"""
    out = []
    top = min(d_max(p), cfg.get("max_degree_B", 99))
    for d in range(1, top + 1):
        out.append(("poly", f"d{d}", comb(p + d, d) - 1))
    for m in cfg["fourier_mults"]:
        out.append(("fourier", f"D{p * m}", p * m))
        out.append(("rff", f"D{p * m}", p * m))
    return out


def basis_matrix(Z, family, D, p):
    """features of the inputs Z for one fixed basis (label-free)"""
    if family == "poly":
        d = next(d for d in range(1, 40) if comb(p + d, d) - 1 == D)
        return hermite_design(Z, d)[:, 1:]                       # drop the constant (intercept is separate)
    if family == "fourier":
        from aqfm_prescreen.dictionaries import balanced_fourier
        return balanced_fourier(Z, D)
    if family == "rff":
        from sklearn.kernel_approximation import RBFSampler
        return RBFSampler(gamma=1.0 / p, n_components=D, random_state=0).fit(Z).transform(Z)
    raise ValueError(family)


def _rows_out(names, learner, config, size, n, res, extra_key=None):
    out = []
    for m, nm in enumerate(names):
        r = dict(name=nm, learner=learner, config=config, size=size, n=n,
                 val_loss=float(res["val_loss"][m]), val_auc=float(res["val_auc"][m]),
                 test_auc=float(res["test_auc"][m]))
        if extra_key:
            r[extra_key] = float(np.asarray(res[extra_key])[m])
        out.append(r)
    return out


def _write(f, rows):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(f)


def gpu_partB_job(args):
    """one GPU job: (kind, p, ...) for every n of the grid; each n is one output file, skipped if
    it exists (restartable)"""
    what, p, spec, smoke = args
    cfg, out = partB_cfg(smoke), partB_dir(smoke)
    import torch                                                  # only GPU workers import torch
    from studies.common import torchlearn as tl
    torch.set_num_threads(1)                                      # one CPU thread per GPU worker
    dev = os.environ.get(bs.TORCH_ENV, "cpu")
    t0 = time.time()
    if what == "mlp":
        w, dep = spec
        files = {n: out / f"mlp_p{p}_w{w}_d{dep}_n{n}.csv" for n in cfg["n_grid"]}
    elif what == "basis":
        fam, lab, D = spec
        files = {n: out / f"basis_p{p}_{fam}_{lab}_n{n}.csv" for n in cfg["n_grid"]}
    else:
        files = {n: out / f"ref_p{p}_n{n}.csv" for n in cfg["n_grid"]}
    todo = [n for n, f in files.items() if not f.exists()]
    label = f"{what} p{p} {spec}"
    if not todo:
        return "skip", label
    Y, names = load_teachers(p)
    Z = bs.pool_inputs(p)[:n_rows()]
    torch.backends.cuda.matmul.allow_tf32 = what == "mlp"         # TF32 matmuls for networks only;
    torch.backends.cudnn.allow_tf32 = what == "mlp"               # logistic regressions stay full fp32
    if what == "basis":
        Xall = basis_matrix(Z, fam, D, p).astype(np.float32)
    for n in todo:
        tr, va, te = splits(n, cfg["val_frac"])
        if what == "mlp":
            res = tl.train_mlps(Z[tr], Y[tr], Z[va], Y[va], Z[te], Y[te], w, dep, dev,
                                seed=_seed(f"s1b_mlp_p{p}_w{w}_d{dep}_n{n}"), **cfg["mlp"])
            rows = _rows_out(names, "mlp", f"w{w}_d{dep}", res["params"], n, res, "best_step")
        elif what == "basis":
            res = tl.logreg_path(Xall[tr], Y[tr], Xall[va], Y[va], Xall[te], Y[te], cfg["lams"], dev)
            rows = _rows_out(names, fam, lab, D, n, res, "lam")
        else:
            rows = []
            specs = teacher_specs(p)
            for key in bank_keys(p):
                idx = [k for k, t in enumerate(specs) if (t["kind"], t["c"], t["tau"]) == key]
                B = bs.load_rows(BANKS, p, *key, DESIGN["n_blocks"])
                X = family_matrix(B, "XYZ", p).astype(np.float32)
                res = tl.logreg_path(X[tr], Y[tr][:, idx], X[va], Y[va][:, idx], X[te], Y[te][:, idx],
                                     cfg["lams"], dev)
                rows += _rows_out([names[k] for k in idx], "ref", "own_XYZ", X.shape[1], n, res, "lam")
        _write(files[n], rows)
    return f"ok {time.time() - t0:.0f}s", label


def _gpu_partB_jobs(cfg, smoke):
    jobs = []
    for p in cfg["ps"]:
        jobs.append(("ref", p, None, smoke))
        for b in basis_sizes(p, cfg):
            jobs.append(("basis", p, b, smoke))
        for w in cfg["mlp_widths"]:
            for dep in cfg["mlp_depths"]:
                jobs.append(("mlp", p, (w, dep), smoke))
    cost = lambda j: (j[2][0] ** 2 * j[2][1] if j[0] == "mlp" else (j[2][2] if j[0] == "basis" else 1))
    jobs.sort(key=lambda j: -cost(j))                              # biggest first
    return jobs


def do_partB_gpu(gpus, smoke):
    if gpus < 1 or gpus > 31:
        raise SystemExit("STOP: --gpus must be 1-31")
    cfg = partB_cfg(smoke)
    jobs = _gpu_partB_jobs(cfg, smoke)
    print(f"{len(jobs)} GPU jobs on {gpus} GPU(s), p = {cfg['ps']}, n = {cfg['n_grid']}", flush=True)
    ctx = mp.get_context("spawn")
    idq = ctx.Queue()
    for i in range(gpus):
        idq.put(i)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(gpus, mp_context=ctx, initializer=_gpu_init, initargs=(idq,)) as ex:
        futs = {ex.submit(gpu_partB_job, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                st, nm = fu.result()
                print(f"[{k}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{k}/{len(jobs)}] FAILED {futs[fu][:3]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun the same command to retry (finished files are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def gbt_job(args):
    """gradient-boosted trees (scikit-learn HistGradientBoosting, one thread) for a chunk of
    teachers at one (p, n, leaves); one fit of max(rounds) rounds, read out at every rounds value"""
    p, n, leaves, lo, hi, smoke = args
    cfg, out = partB_cfg(smoke), partB_dir(smoke)
    f = out / f"gbt_p{p}_L{leaves}_n{n}_t{lo}-{hi}.csv"
    label = f"gbt p{p} n{n} L{leaves} teachers {lo}-{hi}"
    if f.exists():
        return "skip", label
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    Y, names = load_teachers(p)
    Z = bs.pool_inputs(p)[:n_rows()]
    tr, va, te = splits(n, cfg["val_frac"])
    rounds = sorted(cfg["gbt_rounds"])
    Xev = np.vstack([Z[va], Z[te]])
    rows = []
    for m in range(lo, min(hi, len(names))):
        clf = HistGradientBoostingClassifier(max_iter=rounds[-1], max_leaf_nodes=leaves,
                                             learning_rate=cfg["gbt_lr"], early_stopping=False,
                                             random_state=0)
        clf.fit(Z[tr], Y[tr, m].astype(int))
        yv, yt = Y[va, m], Y[te, m]
        for r, s in enumerate(clf.staged_decision_function(Xev), 1):
            if r not in rounds:
                continue
            sv, st = s[:len(va)], s[len(va):]
            vl = float(np.mean(np.logaddexp(0, sv) - yv * sv))       # mean BCE on validation
            ok_v, ok_t = 0 < yv.sum() < len(yv), 0 < yt.sum() < len(yt)
            rows.append(dict(name=names[m], learner="gbt", config=f"L{leaves}_R{r}", size=r * leaves,
                             n=n, val_loss=vl,
                             val_auc=roc_auc_score(yv, sv) if ok_v else np.nan,
                             test_auc=roc_auc_score(yt, st) if ok_t else np.nan, rounds=r))
    _write(f, rows)
    return f"ok {time.time() - t0:.0f}s", label


def do_partB_trees(workers, smoke):
    cfg = partB_cfg(smoke)
    jobs = []
    for p in cfg["ps"]:
        M = len(teacher_specs(p))
        for n in cfg["n_grid"]:
            for L in cfg["gbt_leaves"]:
                for lo in range(0, M, cfg["gbt_chunk"]):
                    jobs.append((p, n, L, lo, lo + cfg["gbt_chunk"], smoke))
    jobs.sort(key=lambda j: -(j[1] * j[2]))                        # biggest first
    print(f"{len(jobs)} tree jobs on {workers} CPU workers", flush=True)
    t0, fails = time.time(), 0
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(gbt_job, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                st, nm = fu.result()
                if st != "skip" and (k % 10 == 0 or k == len(jobs) or smoke):
                    print(f"[{k}/{len(jobs)} {time.time() - t0:6.0f}s] {st} {nm}", flush=True)
            except Exception:
                fails += 1
                print(f"[{k}/{len(jobs)}] FAILED {futs[fu]}\n{traceback.format_exc()}", flush=True)
    print(f"done, {fails} failed" + ("; rerun the same command to retry (finished files are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Part B scoring
# ---------------------------------------------------------------------------

FAMILY_GROUPS = {"fixed": ["poly", "fourier", "rff"], "learned": ["mlp", "gbt"]}


def pick_by_validation(P, learners):
    """for each (teacher, n): the row with the lowest validation loss among `learners`"""
    sub = P[P.learner.isin(learners) & np.isfinite(P.val_loss)]
    return sub.loc[sub.groupby(["name", "n"]).val_loss.idxmin()]


def selected_table(P):
    """one row per (teacher, n, choice): the reference, each family, and the two groups, each
    picked on validation; with the reference's test AUC and the shortfall"""
    ref = P[P.learner == "ref"][["name", "n", "test_auc"]].rename(columns={"test_auc": "ref_auc"})
    parts = []
    choices = {f: [f] for f in sorted(set(P.learner) - {"ref"})}
    choices.update({g: fs for g, fs in FAMILY_GROUPS.items() if set(fs) & set(P.learner)})
    for ch, fs in choices.items():
        s = pick_by_validation(P, fs)[["name", "n", "learner", "config", "size", "test_auc"]]
        s.insert(0, "choice", ch)
        parts.append(s)
    S = pd.concat(parts).merge(ref, on=["name", "n"])
    S["shortfall"] = S.ref_auc - S.test_auc
    S["recovery"] = (S.test_auc - 0.5) / (S.ref_auc - 0.5)
    return S


def cell_shortfall(S, choice, n, p, kind, region=None, cell=None):
    """median shortfall over seeds, per cell (c, tau)"""
    q = S[(S.choice == choice) & (S.n == n) & (S.p == p) & (S.kind == kind)]
    if region is not None:
        q = q[q.region == region]
    if cell is not None:
        q = q[(q.c == cell[0]) & (q.tau == cell[1])]
    return q.groupby(["c", "tau"]).shortfall.median()


def checks_partB(S, cfg=PARTB):
    tol, n, ps = cfg["gets_there"], cfg["test_n"], cfg["ps"]
    rows = []
    # B0 (sanity): learned models get there in the short-anneal control, exact teachers, every p
    v = {p: cell_shortfall(S, "learned", n, p, "Q", cell=(4.0, 1.0)) for p in ps}
    worst = max(float(x.max()) for x in v.values())
    rows.append(("B0", "sanity: learned models get there in the short-anneal control (c=4, tau=1), exact "
                 "teachers, every p", f"largest shortfall {worst:.3f}", worst <= tol))
    # B1: fixed bases short by more than tol in >= 9 of 12 window cells at every p
    cnt = {p: int((cell_shortfall(S, "fixed", n, p, "Q", region="window") > tol).sum()) for p in ps}
    rows.append(("B1", "fixed bases short by > 0.01 in >= 9 of 12 window cells (exact), every p",
                 "; ".join(f"p{p}: {c}/12" for p, c in cnt.items()), min(cnt.values()) >= 9))
    # B2: at p = 8, learned beats fixed in >= 9 of 12 window cells, and gets there in <= 5
    tp = cfg["test_p"]
    lf = cell_shortfall(S, "learned", n, tp, "Q", region="window")
    ff = cell_shortfall(S, "fixed", n, tp, "Q", region="window")
    better = int((lf < ff).sum())
    there = int((lf <= tol).sum())
    rows.append(("B2", f"p = {tp}: learned smaller shortfall than fixed in >= 9 of 12 window cells AND gets "
                 "there in <= 5 of 12", f"better in {better}/12; gets there in {there}/12",
                 better >= 9 and there <= 5))
    # B3: growth with p (exact window slope > 0) and not quantum-specific (exact - MF contains 0)
    q = S[(S.choice == "learned") & (S.n == n) & (S.region == "window") & S.p.isin(ps)]
    U = []
    for key, g in q.groupby(["kind", "c", "tau", "seed"]):
        if g.p.nunique() >= 3:
            U.append(dict(kind=key[0], slope=float(np.polyfit(g.p, g.shortfall, 1)[0])))
    U = pd.DataFrame(U)
    rng = np.random.default_rng(_seed("s1_scoreB"))
    nb = DESIGN["n_boot"]
    bq = _boot_median(U[U.kind == "Q"].slope, rng, nb)
    bm = _boot_median(U[U.kind == "MF"].slope, rng, nb)
    mq, mm = float(U[U.kind == "Q"].slope.median()), float(U[U.kind == "MF"].slope.median())
    lo, hi = _ci(bq)
    rows.append(("B3a", "growth: median slope of the learned shortfall against p (exact, window) > 0",
                 f"{mq:.4f} [{lo:.4f}, {hi:.4f}] per qubit", lo > 0))
    lo, hi = _ci(bq - bm)
    rows.append(("B3b", "not quantum-specific: exact minus mean-field slope (window), interval contains 0",
                 f"{mq - mm:.4f} [{lo:.4f}, {hi:.4f}]", lo <= 0 <= hi))
    return pd.DataFrame(rows, columns=["check", "what", "value", "held"])


def n_star(S, choice, tol):
    """smallest n at which a teacher's `choice` gets there (inf if never)"""
    out = []
    for nm, g in S[S.choice == choice].groupby("name"):
        ok = g[g.shortfall <= tol].n
        out.append(dict(name=nm, n_star=float(ok.min()) if len(ok) else np.inf))
    return pd.DataFrame(out)


def do_scoreB():
    cfg = PARTB
    files = sorted(partB_dir().glob("*.csv"))
    if not files:
        raise SystemExit("STOP: no Part B files found")
    P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    M = pd.read_csv(OUT / "S1_TEACHERS.csv")
    S = selected_table(P).merge(M[["name", "p", "kind", "c", "tau", "seed", "region"]], on="name")
    S.to_csv(OUT / "S1_PARTB.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    S["group"] = S.kind + ", " + np.where(S.region == "window", "window", S.region)
    n = cfg["test_n"]
    print(f"\nmedian shortfall (reference AUC minus learner AUC; 0 = matches the open-book student), "
          f"n = {n}, each family's size picked on validation:")
    for ch in ["fixed", "learned", "poly", "fourier", "rff", "mlp", "gbt"]:
        t = S[(S.choice == ch) & (S.n == n)].groupby(["group", "p"]).shortfall.median().unstack("p")
        if len(t):
            print(f"\n{ch}:")
            print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nmedian shortfall of the best learned model by n (exact, window):")
    t = S[(S.choice == "learned") & (S.kind == "Q") & (S.region == "window")].groupby(["n", "p"]).shortfall.median().unstack("p")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nreference (open-book) test AUC, median, by n (exact, window):")
    t = S[(S.choice == "learned") & (S.kind == "Q") & (S.region == "window")].groupby(["n", "p"]).ref_auc.median().unstack("p")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nwhich learned family is picked (share of teachers), n = 30,000:")
    t = S[(S.choice == "learned") & (S.n == n)].groupby(["group", "learner"]).size().unstack(fill_value=0)
    print(t.to_string())
    N = n_star(S, "learned", cfg["gets_there"]).merge(M[["name", "p", "kind", "region"]], on="name")
    N["group"] = N.kind + ", " + np.where(N.region == "window", "window", N.region)
    print("\nB4 (reported): n* = smallest n at which the best learned model gets there (median; inf = never "
          f"up to {max(cfg['n_grid'])}); share of teachers that get there at some n:")
    t = N.groupby(["group", "p"]).n_star.median().unstack("p")
    print(t.to_string(float_format=lambda v: f"{v:.0f}"))
    t = N.assign(ok=np.isfinite(N.n_star)).groupby(["group", "p"]).ok.mean().unstack("p")
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    K = checks_partB(S, cfg)
    K.to_csv(OUT / "S1_PARTB_CHECKS.csv", index=False)
    print(f"\n=== Part B checks (n = {n}) ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S1_PARTB.csv'} and {OUT / 'S1_PARTB_CHECKS.csv'}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["banks", "teachers", "partA", "scoreA",
                                      "partB_gpu", "partB_trees", "scoreB"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ps", default=",".join(map(str, DESIGN["ps"])))
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--cpu-workers", type=int, default=29)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.workers > 31 or a.cpu_workers > 31:
        raise SystemExit("STOP: at most 31 workers (32-core limit)")
    OUT.mkdir(parents=True, exist_ok=True)
    ps = [int(x) for x in a.ps.split(",")]
    if a.phase == "banks":
        do_banks(ps, a.gpus, a.cpu_workers)
    elif a.phase == "teachers":
        do_teachers(ps, a.workers)
    elif a.phase == "partA":
        do_partA(ps, a.workers)
    elif a.phase == "scoreA":
        do_scoreA()
    elif a.phase == "partB_gpu":
        do_partB_gpu(a.gpus, a.smoke)
    elif a.phase == "partB_trees":
        do_partB_trees(a.workers, a.smoke)
    else:
        do_scoreB()


if __name__ == "__main__":
    main()
