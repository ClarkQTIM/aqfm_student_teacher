#!/usr/bin/env python3
"""
Check that THIS code regenerates the saved tau-sweep results (the results of
record behind docs/aqfm_overview.tex).  Nothing in the run folder is modified:
everything is recomputed into  _verify/  next to this repo.

Checks
  1. banks     re-simulate a sample of saved feature banks (exact, mean-field,
               J=0, ground state; every p) and compare to the saved .npy files
  2. labels    rebuild the labels of a sample of datasets from the regenerated
               banks and compare to the saved labels
  3. scoring   re-run the score steps (ts_tau_sweep score, ts_addons score) on
               the saved per-dataset runs and compare the tables to the saved
               TT_DATASETS.csv and ADDONS_DATASETS.csv
  4. --full    additionally re-run the complete student pipeline for one p=4
               dataset (~5 min) and compare every fold's AUC to the saved run

Usage:
  python tools/verify_reproduction.py --run runs/tau_sweep [--full]
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import io
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import teacher_student as ts            # noqa: E402
import ts_tau_sweep as tt               # noqa: E402
import ts_addons as ad                  # noqa: E402
from aqfm_prescreen.aqfm import compute_bank   # noqa: E402

BANK_TOL = 1e-6        # banks are stored from float32 simulations
AUC_TOL = 1e-9
results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def regen_bank(p, kind, kappa, tau):
    _, Z = tt.inputs(p)
    J = 0.0 if kind == "J0" else kappa / max(p - 1, 1)
    if kind == "GS":
        return np.asarray(tt.ground_state_bank(Z, J), dtype=np.float64)
    return np.asarray(compute_bank(Z, J, tau, method="meanfield" if kind == "MF" else "strang"), dtype=np.float64)


def check_banks(run):
    print("\n== 1. feature banks ==")
    sample = [(p, k, kap, tau) for p in (4, 6, 8)
              for (k, kap, tau) in (("Q", 4.0, 5.0), ("Q", 2.0, 10.0), ("MF", 2.0, 5.0), ("J0", 0.0, 1.0), ("GS", 2.0, 0.0))]
    regen = {}
    for p, k, kap, tau in sample:
        saved = tt.load_bank(run, p, k, kap, tau)
        new = regen_bank(p, k, kap, tau)
        regen[(p, k, kap, tau)] = new
        d = float(np.abs(saved - new).max()) if saved.shape == new.shape else np.inf
        report(f"bank p={p} {tt.bank_key(k, kap, tau)}", d <= BANK_TOL, f"(max |diff| = {d:.2e}, shape {new.shape})")
    return regen


def check_labels(run, regen):
    print("\n== 2. labels from regenerated banks ==")
    specs = {s["name"]: s for s in tt.teacher_specs()}
    names = ["tt_Q_p8_k4_t5_local_s0", "tt_Q_p6_k2_t10_zlocal_s0", "tt_MF_p4_k2_t5_local_s0",
             "tt_J0_p6_k0_t1_local_s1", "tt_Q_p4_k4_t5_zlocal_s0"]
    orig = tt.load_bank
    for nm in names:
        sp = specs[nm]
        key = (sp["p"], sp["kind"], sp["kappa"], sp["tau"])
        if key not in regen:
            regen[key] = regen_bank(*key)

        def patched(out, p, kind, kappa, tau, _k=key):
            return regen[_k] if (p, kind, kappa, tau) == _k else orig(out, p, kind, kappa, tau)
        tt.load_bank = patched
        try:
            y, meta = tt.make_labels(run, sp)
        finally:
            tt.load_bank = orig
        saved = np.load(run / "labels" / f"{nm}.npy")
        report(f"labels {nm}", np.array_equal(y, saved), f"({int((y != saved).sum())} of {len(y)} differ)")


def _link(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def _compare_tables(new, saved, label):
    A, B = pd.read_csv(new), pd.read_csv(saved)
    if len(A) != len(B) or set(A.columns) != set(B.columns):
        report(label, False, f"(shape {A.shape} vs {B.shape})")
        return
    A, B = A.sort_values("name").reset_index(drop=True), B.sort_values("name").reset_index(drop=True)[A.columns]
    bad = []
    for c in A.columns:
        a, b = A[c], B[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-12, equal_nan=True):
                bad.append(c)
        elif not (a.astype(str) == b.astype(str)).all():
            bad.append(c)
    report(label, not bad, f"({len(A)} rows x {len(A.columns)} columns" + (f"; differing: {bad}" if bad else "") + ")")


def check_scoring(run, work):
    print("\n== 3. scoring from the saved per-dataset runs ==")
    sc = work / "rescore"
    if sc.exists():
        shutil.rmtree(sc)
    sc.mkdir(parents=True)
    for nm in ("runs", "labels", "banks", "addons", "TT_TEACHERS.csv", "TT_PROTOCOL.json"):
        _link(run / nm, sc / nm)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        tt.do_score(sc)
    (work / "rescore_tt.txt").write_text(buf.getvalue())
    _compare_tables(sc / "TT_DATASETS.csv", run / "TT_DATASETS.csv", "TT_DATASETS.csv")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        ad.do_score(sc, "TT_TEACHERS.csv", "TT_DATASETS.csv")
    (work / "rescore_addons.txt").write_text(buf.getvalue())
    _compare_tables(sc / "ADDONS_DATASETS.csv", run / "ADDONS_DATASETS.csv", "ADDONS_DATASETS.csv")


def check_full(run, work, nm="tt_Q_p4_k2_t5_local_s0"):
    print("\n== 4. full student pipeline, one dataset (slow) ==")
    p = int(nm.split("_p")[1].split("_")[0])
    S, C = tt._student_sets(run, p)
    y = np.load(run / "labels" / f"{nm}.npy")
    R, _ = ts.run_dataset(tt._CACHE["X"], y, {"name": nm}, tt.DESIGN, S=S, C=C)
    saved = pd.read_csv(run / "runs" / f"{nm}.csv")
    m = R.merge(saved, on=["fold", "arm"], suffixes=("", "_saved"))
    d = float((m.auc - m.auc_saved).abs().max())
    same = (m.choice.astype(str) == m.choice_saved.astype(str)).all()
    report(f"student run {nm}", len(m) == len(saved) and d <= AUC_TOL and same,
           f"(max |AUC diff| = {d:.1e}; same choices: {same})")
    R.to_csv(work / f"{nm}_rerun.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "tau_sweep")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--full-dataset", default="tt_Q_p4_k2_t5_local_s0")
    a = ap.parse_args()
    run = a.run.resolve()
    work = ROOT / "_verify"
    work.mkdir(exist_ok=True)
    import scipy
    import sklearn
    print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}")
    print(f"run folder: {run}\nscratch output: {work}")
    regen = check_banks(run)
    check_labels(run, regen)
    check_scoring(run, work)
    if a.full:
        check_full(run, work, a.full_dataset)
    print(f"\n{sum(results)}/{len(results)} checks pass")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
