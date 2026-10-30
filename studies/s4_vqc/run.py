#!/usr/bin/env python3
"""
S4: does a variational quantum circuit reach the exact map's performance with few parameters?
Plan and predictions: studies/s4_vqc/PLAN.md (fixed before this code ran on real data).

Students: G1 / G2 (generic, inputs encoded once / every layer), H1 / H2 (Hamiltonian-informed,
shared / per-qubit), L = 1, 2, 4, 8, 16 layers, every S1 teacher at p = 4, 6, 8.  Same rows and
splits as S1 Part B, so the open-book student and Part B's best classical learner compare row
for row (runs/studies/s1/S1_PARTB.csv).

Phases
  time    the slowest job (G1, p = 8, L = 16, n = 30,000) for 200 steps, and an estimate for all
  train   on the GPUs (--gpus N): one job per (student, L, p, n), restartable
            -> runs/studies/s4/train/{kind}_L{L}_p{p}_n{n}.csv
  score   each student picks L on validation; checks V0-V4
            -> runs/studies/s4/S4_SELECTED.csv, S4_CHECKS.csv
  --smoke tiny version at p = 4 to runs/studies/s4_smoke/ (timings only)

Usage (from the repo root):
  python studies/s4_vqc/run.py time  --gpus 2
  python studies/s4_vqc/run.py train --gpus 2
  python studies/s4_vqc/run.py score
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studies.common import bankstore as bs     # noqa: E402

_spec = importlib.util.spec_from_file_location("s1run_for_s4", ROOT / "studies" / "s1_scaling" / "run.py")
S1 = importlib.util.module_from_spec(_spec)
sys.modules["s1run_for_s4"] = S1
_spec.loader.exec_module(S1)       # teachers, splits, GPU pool, GPU check

DESIGN = {
    "version": "s4_1",
    "plan": "studies/s4_vqc/PLAN.md",
    "kinds": ["G1", "G2", "H1", "H2"],
    "Ls": [1, 2, 4, 8, 16],
    "ps": [4, 6, 8],
    "n_main": 3_000,
    "n_big": 30_000, "p_big": 8,
    "train": dict(lr=0.01, batch=256, eval_every=25, patience=40, max_steps=10_000),
    "test_p": 8,
    "gets_there": 0.01,
}
SMOKE = dict(DESIGN, kinds=["G1", "G2", "H1", "H2"], Ls=[1, 2], ps=[4], n_main=300, n_big=None,
             train=dict(lr=0.01, batch=256, eval_every=10, patience=3, max_steps=60))
OUT = Path("runs/studies/s4")


def cfg(smoke):
    return SMOKE if smoke else DESIGN


def out_dir(smoke):
    return Path("runs/studies/s4_smoke") if smoke else OUT


def jobs(c, smoke):
    J = [(k, L, p, c["n_main"], smoke) for k in c["kinds"] for L in c["Ls"] for p in c["ps"]]
    if c.get("n_big"):
        J += [(k, L, c["p_big"], c["n_big"], smoke) for k in c["kinds"] for L in c["Ls"]]
    cost = lambda j: (2 ** j[2]) * j[1] * (j[2] if j[0] in ("G1", "G2", "H2") else 1) * (j[3] ** 0.3)
    return sorted(J, key=lambda j: -cost(j))


def train_job(args):
    kind, L, p, n, smoke = args
    c = cfg(smoke)
    f = out_dir(smoke) / "train" / f"{kind}_L{L}_p{p}_n{n}.csv"
    label = f"{kind} L{L} p{p} n{n}"
    if f.exists():
        return "skip", label
    import torch
    from studies.common import torchvqc as tv
    torch.set_num_threads(1)
    dev = os.environ.get(bs.TORCH_ENV, "cpu")
    t0 = time.time()
    Y, names = S1.load_teachers(p)
    Z = bs.pool_inputs(p)[:S1.n_rows()]
    tr, va, te = S1.splits(n, S1.PARTB["val_frac"])
    res = tv.train_circuits(kind, L, Z[tr], Y[tr], Z[va], Y[va], Z[te], Y[te], dev,
                            seed=S1._seed(f"s4_{kind}_L{L}_p{p}_n{n}"), **c["train"])
    rows = [dict(name=nm, kind=kind, L=L, p=p, n=n, params=res["params"], val_loss=float(res["val_loss"][m]),
                 val_auc=float(res["val_auc"][m]), test_auc=float(res["test_auc"][m]),
                 best_step=int(res["best_step"][m]), steps=int(res["steps"])) for m, nm in enumerate(names)]
    S1._write(f, rows)
    return f"ok {time.time() - t0:.0f}s ({res['steps']} steps)", label


def do_train(gpus, smoke):
    if gpus < 1 or gpus > 31:
        raise SystemExit("STOP: --gpus must be 1-31")
    S1.require_gpus(gpus)
    J = jobs(cfg(smoke), smoke)
    print(f"{len(J)} circuit-training jobs on {gpus} GPU(s)", flush=True)
    fails = S1._run_gpu_pool(J, train_job, gpus)
    print(f"done, {fails} failed" + ("; rerun the same command (finished files are skipped)" if fails else ""))
    if fails:
        sys.exit(1)


def time_job(args):
    import torch
    from studies.common import torchvqc as tv
    dev = os.environ.get(bs.TORCH_ENV, "cpu")
    kind, L, p = args
    Y, _ = S1.load_teachers(p)
    Z = bs.pool_inputs(p)[:4000]
    t0 = time.time()
    tv.train_circuits(kind, L, Z[:3000], Y[:3000], Z[3000:3500], Y[3000:3500], Z[3500:4000], Y[3500:4000], dev,
                      seed=0, lr=0.01, batch=256, eval_every=1000, patience=10 ** 9, max_steps=200)
    return f"{(time.time() - t0) / 200 * 1e3:.1f} ms/step", f"{kind} L{L} p{p}"


def do_time(gpus):
    S1.require_gpus(1)
    probes = [("G1", 16, 8), ("H2", 16, 8), ("H1", 16, 8), ("G1", 4, 6)]
    for pr in probes:
        S1._run_gpu_pool([pr], time_job, 1)
    n_jobs = len(jobs(DESIGN, False))
    print(f"(time per training step, 99 circuits at once.  {n_jobs} jobs, each up to "
          f"{DESIGN['train']['max_steps']} steps, usually fewer with early stopping)")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def cell_median(T, col, kind, p, n, region="window", cell=None):
    q = T[(T.kind == kind) & (T.p == p) & (T.n == n) & (T.teacher_kind == "Q")]
    q = q[q.region == region] if cell is None else q[(q.c == cell[0]) & (q.tau == cell[1])]
    return q.groupby(["c", "tau"])[col].median()


def do_score():
    c = DESIGN
    files = sorted((OUT / "train").glob("*.csv"))
    if not files:
        raise SystemExit("STOP: no S4 training files")
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    M = pd.read_csv(S1.OUT / "S1_TEACHERS.csv").rename(columns={"kind": "teacher_kind", "p": "tp"})
    B = pd.read_csv(S1.OUT / "S1_PARTB.csv")
    ref = B[B.choice == "learned"][["name", "n", "ref_auc", "test_auc"]].rename(columns={"test_auc": "classical_auc"})
    # each student kind picks L on validation loss (per teacher and n)
    sel = R.loc[R.groupby(["name", "kind", "n"]).val_loss.idxmin()]
    # the "better of" groups, also picked on validation
    grp = []
    for g, ks in (("G", ["G1", "G2"]), ("H", ["H1", "H2"])):
        s = sel[sel.kind.isin(ks)].reset_index(drop=True)
        s = s.loc[s.groupby(["name", "n"]).val_loss.idxmin()].assign(kind=g)
        grp.append(s)
    T = pd.concat([sel] + grp, ignore_index=True).merge(ref, on=["name", "n"]).merge(
        M[["name", "teacher_kind", "c", "tau", "seed", "region"]], on="name")
    T["shortfall"] = T.ref_auc - T.test_auc
    T["vs_classical"] = T.test_auc - T.classical_auc
    T.to_csv(OUT / "S4_SELECTED.csv", index=False)
    tp, n, tol = c["test_p"], c["n_main"], c["gets_there"]

    # parity by fixed L (H1 at L = 16 for V0; smallest L with parity for V4)
    RL = R.merge(ref, on=["name", "n"]).merge(M[["name", "teacher_kind", "c", "tau", "region"]], on="name")
    RL["shortfall"] = RL.ref_auc - RL.test_auc
    rows = []
    v0 = {p: float(RL[(RL.kind == "H1") & (RL.L == 16) & (RL.p == p) & (RL.n == n) & (RL.teacher_kind == "Q")
                      & (RL.c == 4.0) & (RL.tau == 1.0)].groupby(["c", "tau"]).shortfall.median().max())
          for p in c["ps"]}
    rows.append(("V0", "sanity: H1 at L = 16 within 0.01 of open-book in the short-anneal control, p = 4, 6, 8",
                 "; ".join(f"p{p}: {v:.3f}" for p, v in v0.items()), all(v <= tol for v in v0.values())))
    g_vs = cell_median(T, "vs_classical", "G", tp, n)
    rows.append(("V1", f"generic no better than classical: better of G1/G2 <= Part B best classical + 0.01 in >= 9 of 12 window cells (p = {tp})",
                 f"{int((g_vs <= tol).sum())}/12", bool((g_vs <= tol).sum() >= 9)))
    h_sf = cell_median(T, "shortfall", "H", tp, n)
    rows.append(("V2", f"informed reaches parity: better of H1/H2 within 0.01 of open-book in >= 6 of 12 window cells (p = {tp})",
                 f"{int((h_sf <= tol).sum())}/12", bool((h_sf <= tol).sum() >= 6)))
    h1 = cell_median(T, "test_auc", "H1", tp, n)
    g = cell_median(T, "test_auc", "G", tp, n)
    rows.append(("V3", f"structure beats generic: H1 AUC > better generic in >= 9 of 12 window cells (p = {tp})",
                 f"{int((h1 > g).sum())}/12", bool((h1 > g).sum() >= 9)))
    K = pd.DataFrame(rows, columns=["check", "what", "value", "held"])
    K.to_csv(OUT / "S4_CHECKS.csv", index=False)

    pd.set_option("display.width", 220)
    T["group"] = T.teacher_kind + ", " + np.where(T.region == "window", "window", T.region)
    print(f"\nmedian shortfall vs the open-book student (n = {n}; each student picks L on validation):")
    print(T[T.n == n].groupby(["group", "kind", "p"]).shortfall.median().unstack("p")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nmedian AUC minus Part B's best classical learner (n = {n}):")
    print(T[T.n == n].groupby(["group", "kind", "p"]).vs_classical.median().unstack("p")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwindow, exact teachers, p = {tp}: median shortfall by L (fixed L, no selection), n = {n}")
    w = RL[(RL.teacher_kind == "Q") & (RL.region == "window") & (RL.p == tp) & (RL.n == n)]
    print(w.groupby(["kind", "L"]).agg(shortfall=("shortfall", "median"), params=("params", "first"))
          .unstack("kind").to_string(float_format=lambda v: f"{v:.3f}"))
    if c.get("n_big"):
        print(f"\nn = {c['n_big']}, p = {c['p_big']}: median shortfall by student")
        print(T[T.n == c["n_big"]].groupby(["group", "kind"]).shortfall.median().unstack("kind")
              .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== S4 checks ===")
    print(K.to_string(index=False))
    print(f"\nwrote {OUT / 'S4_SELECTED.csv'}, {OUT / 'S4_CHECKS.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["time", "train", "score"])
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "time":
        do_time(a.gpus)
    elif a.phase == "train":
        do_train(a.gpus, a.smoke)
    else:
        do_score()


if __name__ == "__main__":
    main()
