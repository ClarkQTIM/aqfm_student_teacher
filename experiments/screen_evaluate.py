"""
Score the screen study's checks (docs/screen_plan.md, Section 6 and Amendments 1-4).

Two parts:
  assemble(out, sweep)  IO: one row per library dataset (1152) with its teacher
                        facts, student AUCs, screen measures, true ceiling and true
                        best accuracy.  -> SCREEN_DATASETS.csv
  checks(D)             pure: the checks P1-P12 from that table.  -> SCREEN_RESULTS.csv

The truths used here (ceiling, best accuracy, "resistant", "AQFM helps") are for
EVALUATION only; screen_measures.py never sees them.

Operational definitions (fixed here before any result is in; Amendment 4):
  best cheap     max AUC of classical_best, nonlinear_best, S_MF, S_J0, S_GS (B2)
  margin         AUC(S_Q) - best cheap  (identical to the overview's margin_new)
  resistant      ceiling - best cheap >= 0.05
  AQFM helps     margin >= 0.05
  A1             max(classical_best, nonlinear_best): best classical AUC, no map
  screen score   mean of the percentile ranks (over all 1152) of L1_slope and L2_gain
  flag           screen score >= a threshold chosen on two p values (maximising
                 TPR - FPR for "resistant") and applied to the third
  cluster        for bootstrap intervals: a teacher and its noisier copy are one
                 cluster (they share w and the random draws); 1000 resamples
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESIST = 0.05
HELPS = 0.05
N_BOOT = 1000
OPPONENTS = ["auc_classical_best", "auc_nonlinear_best", "auc_S_MF", "auc_S_J0", "auc_S_GS"]
ORDER_TRUTH = {"LIN": "1", "PAIR": "2", "TRIPLE": "3", "PARp": ">=4"}


# ---------------------------------------------------------------------------
# assemble (IO)
# ---------------------------------------------------------------------------

def _students(run_dir, name):
    """mean-over-folds AUC of every student arm and nonlinear learner, or {} if missing"""
    out = {}
    f = run_dir / "runs" / f"{name}.csv"
    if f.exists():
        R = pd.read_csv(f)
        out.update({f"auc_{a}": v for a, v in R.groupby("arm").auc.mean().items()})
    g = run_dir / "addons" / "nl" / f"{name}.csv"
    if g.exists():
        N = pd.read_csv(g)
        out.update({f"auc_{a}": v for a, v in N.groupby("arm").auc.mean().items()})
    return out


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _best_accuracy_aqfm(sweep, sp, temp):
    """true best accuracy from the teacher's own probabilities (recomputed exactly
    as make_labels does: same seed, so the same w)"""
    import ts_tau_sweep as tt
    rng = np.random.default_rng(sp["data_seed"])
    F = tt.readout_matrix(tt.load_bank(sweep, sp["p"], sp["kind"], sp["kappa"], sp["tau"]), sp["readout"], sp["p"])
    score = tt._score_from(F, rng.standard_normal(F.shape[1]))
    q = _sigmoid(score / temp)
    return float(np.maximum(q, 1 - q).mean())


def _best_accuracy_foreign(sp):
    import ts_screen as SC
    if sp["kind"] == "NOISE":
        return 0.5                                # labels do not depend on z
    _, score, _ = SC.make_foreign(sp)
    q = _sigmoid(score / sp["temp"])
    return float(np.maximum(q, 1 - q).mean())


def assemble(out, sweep):
    import ts_addons as ad
    import ts_screen as SC
    import ts_tau_sweep as tt
    specs = {s["name"]: s for s in tt.teacher_specs()}
    fspecs = {s["name"]: s for s in SC.foreign_specs()}
    T = pd.read_csv(sweep / "TT_TEACHERS.csv")
    T = T[~T.degenerate.astype(bool)].assign(block="tau_sweep", temp=tt.DESIGN["temp"], ceiling=lambda d: d.oracle_auc)
    T["base"] = T.name
    S = pd.read_csv(out / SC.TEACHERS_FILE)
    S = S[~S.degenerate.astype(bool)]
    keep = ["name", "block", "kind", "p", "kappa", "tau", "readout", "seed", "temp", "base", "ceiling"]
    D = pd.concat([T[keep], S[keep]], ignore_index=True)
    rows = []
    for r in D.itertuples():
        run_dir = sweep if r.block == "tau_sweep" else out
        row = {"name": r.name, **_students(run_dir, r.name)}
        if r.block == "foreign":
            row["best_acc"] = _best_accuracy_foreign(fspecs[r.name])
        else:
            row["best_acc"] = _best_accuracy_aqfm(sweep, specs[r.base], r.temp)
        rows.append(row)
    D = D.merge(pd.DataFrame(rows), on="name", how="left")
    M = out / "SCREEN_MEASURES.csv"
    if M.exists():
        D = D.merge(pd.read_csv(M).drop(columns=["block"], errors="ignore"), on="name", how="left")
    D["region"] = [ad._region(r) if r.block != "foreign" else "foreign" for r in D.itertuples()]
    D = add_derived(D)
    # cross-check against the tau sweep's own scoring
    A = sweep / "ADDONS_DATASETS.csv"
    if A.exists() and "margin" in D:
        a = pd.read_csv(A)[["name", "margin_new"]].merge(D[["name", "margin"]], on="name").dropna()
        d = float((a.margin_new - a.margin).abs().max()) if len(a) else np.nan
        print(f"cross-check: margin vs the tau sweep's ADDONS_DATASETS.csv on {len(a)} datasets, "
              f"max |diff| = {d:.1e}" + ("  OK" if d < 1e-9 else "  <-- MISMATCH"))
    return D


def add_derived(D):
    D = D.copy()
    opp = [c for c in OPPONENTS if c in D]
    if opp:
        D["best_cheap"] = D[opp].max(axis=1, skipna=False)
        # truths are stored as 1.0 / 0.0 / NaN (unknown while runs are unfinished)
        D["resistant"] = np.where(D.best_cheap.notna(), (D.ceiling - D.best_cheap >= RESIST).astype(float), np.nan)
    if "auc_S_Q" in D and "best_cheap" in D:
        D["margin"] = D.auc_S_Q - D.best_cheap
        D["helps"] = np.where(D.margin.notna(), (D.margin >= HELPS).astype(float), np.nan)
    if {"auc_classical_best", "auc_nonlinear_best"} <= set(D):
        D["A1"] = D[["auc_classical_best", "auc_nonlinear_best"]].max(axis=1, skipna=False)
    if {"L1_slope", "L2_gain"} <= set(D):
        D["screen"] = (D.L1_slope.rank(pct=True) + D.L2_gain.rank(pct=True)) / 2
    return D


# ---------------------------------------------------------------------------
# checks (pure)
# ---------------------------------------------------------------------------

def auroc(score, truth):
    s, t = np.asarray(score, float), np.asarray(truth, float)
    ok = ~(np.isnan(s) | np.isnan(t))
    s, t = s[ok], t[ok].astype(int)
    if len(np.unique(t)) < 2:
        return np.nan
    return float(roc_auc_score(t, s))


def boot_ci(D, fn, n=N_BOOT, seed=0):
    """95% interval of fn(D) resampling whole clusters (column `base`)"""
    rng = np.random.default_rng(seed)
    groups = {b: g.index.to_numpy() for b, g in D.groupby("base")}
    keys = np.array(list(groups))
    if len(keys) == 0:                      # nothing finished yet for this subset
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        v = fn(D.loc[np.concatenate([groups[k] for k in pick])])
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def lopo_flags(D, score="screen", truth="resistant"):
    """flag = score >= threshold, the threshold chosen on the OTHER p values to
    maximise TPR - FPR for `truth`.  Returns 1.0 / 0.0, NaN where unknown."""
    flags = pd.Series(np.nan, index=D.index, dtype=float)
    for p in sorted(D.p.unique()):
        tr = D[(D.p != p) & D[score].notna() & D[truth].notna()]
        te = D.index[(D.p == p) & D[score].notna()]
        if tr.empty or tr[truth].astype(bool).nunique() < 2:
            continue
        y = tr[truth].astype(bool).to_numpy()
        best, thr = -np.inf, np.inf
        for c in np.unique(tr[score]):
            f = tr[score].to_numpy() >= c
            j = f[y].mean() - f[~y].mean()
            if j > best:
                best, thr = j, c
        flags.loc[te] = (D.loc[te, score] >= thr).to_numpy().astype(float)
    return flags


def checks(D):
    """every check that the available columns allow; one row per check"""
    R = []

    def add(check, what, value, ci=(np.nan, np.nan), rule="", passed=None, n=None, note=""):
        R.append(dict(check=check, what=what, value=value, ci_lo=ci[0], ci_hi=ci[1], rule=rule,
                      passed=passed, n=n, note=note))

    has = lambda *c: all(x in D and D[x].notna().any() for x in c)
    sweep = D[D.block == "tau_sweep"]
    aqfm = D[D.block != "foreign"]
    win = aqfm[(aqfm.region == "window") & aqfm.kind.isin(["Q", "MF"])]

    # P1: A1 alone, original noise, already-seen data
    if has("A1", "helps"):
        add("P1", "A1 low -> flag, vs 'AQFM helps' (tau sweep, T=0.25)", auroc(-sweep.A1, sweep.helps),
            rule="reported", n=int(sweep.helps.notna().sum()))
        w = win[win.block == "tau_sweep"]
        add("P1", "A1 low ranks exact-window above mean-field-window (AUROC; < 0.5 = wrong way)",
            auroc(-w.A1, w.kind == "Q"), rule="prediction: < 0.5", n=len(w))

    # P2: true ceiling - best cheap as a ranker for 'AQFM helps'
    if has("best_cheap", "helps"):
        f = lambda d: auroc(d.ceiling - d.best_cheap, d.helps)
        v = f(D)
        add("P2", "(true ceiling - best cheap) vs 'AQFM helps', all datasets", v, boot_ci(D, f), ">= 0.9",
            v >= 0.9 if v == v else None, int(D.helps.notna().sum()))

    # P3': the screen score vs 'resistant'
    if has("screen", "resistant"):
        f = lambda d: auroc(d.screen, d.resistant)
        v = f(D)
        add("P3'", "screen score vs 'resistant', all datasets", v, boot_ci(D, f), ">= 0.75",
            v >= 0.75 if v == v else None, int(D.resistant.notna().sum()))
        for p in sorted(D.p.unique()):
            add("P3'", f"  ... p = {p}", auroc(D[D.p == p].screen, D[D.p == p].resistant), rule="reported")
        for blk in ("tau_sweep", "aqfm_noise", "foreign"):
            d = D[D.block == blk]
            add("P3'", f"  ... block {blk}", auroc(d.screen, d.resistant), rule="reported",
                n=int(d.resistant.notna().sum()))
        for c in ("L1_slope", "L2_gain", "L1_asymptote"):
            if has(c):
                add("P3'", f"  ... {c} alone", auroc(D[c] - (D.best_cheap if c == "L1_asymptote" else 0), D.resistant),
                    rule="reported", note="asymptote used as (asymptote - best cheap)" if c == "L1_asymptote" else "")

    # P4: ceiling estimators against the truth
    if has("L1_asymptote"):
        e = (D.L1_asymptote - D.ceiling).abs()
        add("P4", "L1 asymptote vs true ceiling (AUC): median |error|", float(e.median()), rule="<= 0.03",
            passed=bool(e.median() <= 0.03), n=int(e.notna().sum()))
        for p in sorted(D.p.unique()):
            add("P4", f"  ... p = {p}", float(e[D.p == p].median()), rule="reported")
    if has("L3_ceiling_acc", "best_acc"):
        e = (D.L3_ceiling_acc - D.best_acc).abs()
        add("P4", "L3 nearest-neighbour bound vs true best accuracy: median |error|", float(e.median()),
            rule="<= 0.03", passed=bool(e.median() <= 0.03), n=int(e.notna().sum()),
            note="prediction: fails at p = 8 (underestimates)")
        for p in sorted(D.p.unique()):
            add("P4", f"  ... p = {p} (median signed error)", float((D.L3_ceiling_acc - D.best_acc)[D.p == p].median()),
                rule="reported")

    # P5: exact vs mean-field teachers inside the window
    if has("A1", "auc_S_MF") and len(win):
        for lab, d in (("T=0.25", win[win.temp == 0.25]), ("T=1", win[win.temp == 1.0])):
            if d.empty:
                continue
            y = d.kind == "Q"
            v_a = auroc(-d.A1, y)
            add("P5", f"window, {lab}: data alone (A1 low) separates exact from mean-field", v_a,
                rule="<= 0.5", passed=bool(v_a <= 0.5) if v_a == v_a else None, n=len(d))
            if has("screen"):
                add("P5", f"window, {lab}: screen score separates exact from mean-field", auroc(d.screen, y),
                    rule="reported", n=len(d))
            stand_in = d[["auc_S_MF", "auc_S_J0", "auc_S_GS"]].max(axis=1)
            v_b = auroc(-(stand_in - d.A1), y)
            add("P5", f"window, {lab}: + stand-ins (small stand-in gain over A1 -> exact)", v_b,
                rule=">= 0.9", passed=bool(v_b >= 0.9) if v_b == v_b else None, n=len(d))

    add("P6", "C measures (kernel alignment, Huang et al.)", np.nan, rule=">= 0.85", note="not computed yet: needs the C phase")
    add("P7", "reduced-menu pilot", np.nan, rule=">= 85% same call", note="not computed yet: needs the C phase")

    # P8: worth-it fraction at each noise level, on the paired AQFM subset
    if has("helps"):
        bases = set(D[D.block == "aqfm_noise"].base)
        pair = D[D.base.isin(bases)]
        for T in (0.25, 1.0):
            d = pair[pair.temp == T]
            add("P8", f"AQFM helps, paired AQFM subset, T={T:g}", float(d.helps.dropna().astype(bool).mean())
                if d.helps.notna().any() else np.nan, rule="reported", n=int(d.helps.notna().sum()))
            dq = d[(d.kind == "Q") & (d.region == "window")]
            add("P8", f"  ... exact teachers in the window, T={T:g}", float(dq.helps.dropna().astype(bool).mean())
                if dq.helps.notna().any() else np.nan, rule="reported", n=int(dq.helps.notna().sum()))

    # P9: flag rate on NOISE and LIN (thresholds from the other p values)
    if has("screen", "resistant"):
        D = D.assign(flag=lopo_flags(D))
        for fam in ("NOISE", "LIN"):
            d = D[(D.block == "foreign") & (D.kind == fam)]
            v = float(d.flag.dropna().astype(bool).mean()) if d.flag.notna().any() else np.nan
            add("P9", f"flag rate on {fam}", v, rule="<= 0.10", passed=bool(v <= 0.10) if v == v else None, n=len(d))
        d = D[D.flag.notna() & D.resistant.notna()]
        if len(d):
            fl, rs = d.flag.astype(bool), d.resistant.astype(bool)
            add("P9", "  ... overall: share of resistant datasets flagged", float(fl[rs].mean()), rule="reported", n=int(rs.sum()))
            add("P9", "  ... overall: share of flagged datasets that are resistant", float(rs[fl].mean()) if fl.any() else np.nan,
                rule="reported", n=int(fl.sum()))

    # P10 / P11: how often AQFM helps among resistant (truth) and among flagged (screen) datasets
    if has("resistant", "helps"):
        for chk, blk in (("P10", D.block == "foreign"), ("P11", D.block != "foreign")):
            d = D[blk & D.resistant.notna() & D.helps.notna()]
            r = d[d.resistant.astype(bool)]
            v = float(r.helps.astype(bool).mean()) if len(r) else np.nan
            add(chk, f"AQFM helps | resistant, {'non-AQFM' if chk == 'P10' else 'AQFM-made'} teachers", v,
                boot_ci(d, lambda x: float(x[x.resistant.astype(bool)].helps.astype(bool).mean())
                        if x.resistant.astype(bool).any() else np.nan),
                rule="prediction: <= 0.10" if chk == "P10" else "prediction: much higher than P10", n=len(r))
            if "flag" in D:
                fl = d[d.index.isin(D.index[D.flag == 1.0])]
                add(chk, "  ... among FLAGGED datasets instead", float(fl.helps.astype(bool).mean()) if len(fl) else np.nan,
                    rule="reported", n=len(fl))
            if chk == "P10":
                for fam, g in r.groupby("kind"):
                    add(chk, f"  ... {fam}", float(g.helps.astype(bool).mean()), rule="reported", n=len(g))

    # P12: the interaction ladder on teachers whose order we know
    if has("F1_order"):
        d = D[(D.block == "foreign") & D.kind.isin(list(ORDER_TRUTH)) & D.F1_order.notna()]
        ok = d.F1_order.astype(str) == d.kind.map(ORDER_TRUTH)
        v = float(ok.mean()) if len(d) else np.nan
        add("P12", "F1 order correct on LIN/PAIR/TRIPLE/PARp", v, rule=">= 0.75",
            passed=bool(v >= 0.75) if v == v else None, n=len(d))
        for fam in ORDER_TRUTH:
            g = d[d.kind == fam]
            add("P12", f"  ... {fam} (truth {ORDER_TRUTH[fam]})",
                float((g.F1_order.astype(str) == ORDER_TRUTH[fam]).mean()) if len(g) else np.nan, rule="reported",
                n=len(g), note=", ".join(f"{k}:{v}" for k, v in g.F1_order.astype(str).value_counts().items()))
    return pd.DataFrame(R), D


def report(R):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    out = R.copy()
    out["result"] = out.passed.map({True: "PASS", False: "FAIL"}).fillna("")
    out["ci"] = [f"[{a:.3f}, {b:.3f}]" if a == a else "" for a, b in zip(R.ci_lo, R.ci_hi)]
    out["value"] = out.value.map(lambda v: f"{v:.3f}" if v == v else "—")
    out["n"] = out.n.map(lambda v: "" if v is None or v != v else str(int(v)))
    print(out[["check", "what", "value", "ci", "rule", "result", "n", "note"]].to_string(index=False))
