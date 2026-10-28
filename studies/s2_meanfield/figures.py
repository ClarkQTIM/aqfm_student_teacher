#!/usr/bin/env python3
"""
Figures for S2, made only from S2's own outputs (no numbers typed in).

  python studies/s2_meanfield/figures.py         -> studies/s2_meanfield/figures/*.png

Source, in order of preference:
  1. runs/studies/s2/S2_MEASURES.csv (per cell, p = 2-16) and S2_AMEND.csv (p <= 14):
     medians over cells are computed here;
  2. otherwise the median tables printed in runs/s2_p16.log and runs/s2_amend.log
     (the same medians, as the runner printed them).

fig_s2_meanfield_vs_qubits.png    share of the exact labels mean-field can't explain (U), vs p
fig_s2_correlations_vs_qubits.png total pull of the correlations mean-field ignores (C_coll), vs p
"""
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "figures"
REGIONS = ["control: short anneal", "control: weak coupling", "window"]

BLUE, GRAY, GRAY2 = "#2a78d6", "#8a8984", "#b9b8b2"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 14, "lines.linewidth": 2, "lines.markersize": 8,
    "legend.frameon": False,
})


def parse_log_table(path, start_marker):
    """a pandas groupby table printed in a log: rows 'region  p  v1 v2 ...', the region name only
    on its first row"""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start_marker))
    header = lines[i + 1].split()
    rows, region = [], None
    for l in lines[i + 3:]:
        if not l.strip():
            break
        for r in REGIONS:
            if l.startswith(r):
                region, l = r, l[len(r):]
                break
        parts = l.split()
        rows.append([region, int(parts[0])] + [float(x) for x in parts[1:]])
    return pd.DataFrame(rows, columns=["region", "p"] + header)


def load(runs=RUNS, logs=None):
    """median tables (region, p, ...) for the primary measures and the amendment"""
    m, a = runs / "studies" / "s2" / "S2_MEASURES.csv", runs / "studies" / "s2" / "S2_AMEND.csv"
    if logs is None and m.exists() and a.exists():
        med = pd.read_csv(m).groupby(["region", "p"])[["E", "C_pair", "C_ZZ", "C_coll", "U"]].median().reset_index()
        am = pd.read_csv(a).groupby(["region", "p"])[["U_same400", "U_same800", "U_best800"]].median().reset_index()
        return med, am, f"{m.name} and {a.name}"
    logs = logs or (runs / "s2_p16.log", runs / "s2_amend.log")
    med = parse_log_table(logs[0], "median over cells, by region and p:")
    am = parse_log_table(logs[1], "median over cells, by region and p (U =")
    nm = lambda x: re.sub(r"^[0-9a-f]{8}-", "", Path(x).name)       # drop an upload prefix, if any
    return med, am, f"{nm(logs[0])} and {nm(logs[1])}"


def note(fig, text):
    fig.text(0.01, 0.01, textwrap.fill(text, 150), fontsize=9, color=INK2, va="bottom")


def main(med, am, source, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    w, wc, sc = (med[med.region == r] for r in ("window", "control: weak coupling", "control: short anneal"))
    wa = am[am.region == "window"]
    ps = sorted(med.p.unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(w.p, w.U, "o-", color=BLUE, label="window: mean-field at the same setting")
    ax.plot(wa.p, wa.U_best800, "o--", color=BLUE, mfc=SURF, label="window: best of 60 mean-field settings")
    ax.plot(wc.p, wc.U, "s-", color=GRAY, label="outside: weak coupling (c = 0.5, τ = 5)")
    ax.plot(sc.p, sc.U, "^-", color=GRAY2, label="outside: short anneal (c = 4, τ = 1)")
    ax.set_yscale("log")
    ax.set_xticks(ps)
    ax.set_xlim(min(ps) - 0.5, max(ps) + 1.2)
    ax.set_xlabel("Number of qubits p")
    ax.set_ylabel("Share of the exact labels mean-field gets wrong (U)\nlower = mean-field closer to exact")
    ax.set_title("Outside the window mean-field catches up; inside it, it never does")
    ax.annotate(f"{w.U.iloc[-1]:.2f} at p = {w.p.iloc[-1]}", (w.p.iloc[-1], w.U.iloc[-1]),
                xytext=(0, -38), textcoords="offset points", ha="center", color=INK2)
    ax.annotate(f"{wc.U.iloc[-1]:.4f}", (wc.p.iloc[-1], wc.U.iloc[-1]), xytext=(10, 0),
                textcoords="offset points", ha="left", va="center", color=INK2)
    ax.legend(loc="lower left")
    note(fig, "Log scale: each gridline is 10x the one below. Median over the 12 window settings (c = 2, 4, 8 × "
         "τ = 3, 5, 10, 20) and per outside setting; 400 inputs per point (best-of-60: 800 inputs, up to 14 qubits). "
         "U = share of the exact teacher's noise-free label function that a cross-validated straight-line fit on "
         f"mean-field's outputs can't reproduce (20 random readouts per setting). Source: {source}.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig_s2_meanfield_vs_qubits.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(w.p, w.C_coll, "o-", color=BLUE, label="window")
    ax.plot(wc.p, wc.C_coll, "s-", color=GRAY, label="outside: weak coupling")
    ax.plot(sc.p, sc.C_coll, "^-", color=GRAY2, label="outside: short anneal")
    ax.set_yscale("log")
    ax.set_xticks(ps)
    ax.set_xlabel("Number of qubits p")
    ax.set_ylabel("Pull of the ignored correlations on one qubit\nC_coll = J · |Σ_j (⟨Z_i Z_j⟩ − ⟨Z_i⟩⟨Z_j⟩)|")
    ax.set_title("Why: in the window the correlations mean-field ignores add up instead of fading")
    ax.legend(loc="lower left")
    note(fig, f"Median over settings, exact simulation, 400 inputs. Mean-field assumes this pull is ~0. Source: {source}.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig_s2_correlations_vs_qubits.png", dpi=200)
    plt.close(fig)
    print("wrote", sorted(str(p) for p in out.glob("fig_s2_*.png")), "from", source)


if __name__ == "__main__":
    logs = tuple(sys.argv[1:3]) if len(sys.argv) >= 3 else None
    main(*load(logs=logs))
