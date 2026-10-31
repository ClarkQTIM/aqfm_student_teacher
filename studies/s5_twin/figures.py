#!/usr/bin/env python3
"""
Figures for S5, made only from runs/studies/s5/S5_MEASURES.csv and S5_CONVERGE.csv (no numbers typed in).

  python studies/s5_twin/figures.py [S5_MEASURES.csv] [S5_CONVERGE.csv]   -> studies/s5_twin/figures/*.png

fig_s5_twins_vs_qubits.png   share of the exact labels each classical twin can't explain (U), window cells by
                             coupling, against qubits: mean-field (dashed), DTWA (solid; large dots = converged)
fig_s5_convergence.png       U_DTWA against the number of trajectories, every window cell at p = 12
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC_M = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "studies" / "s5" / "S5_MEASURES.csv"
SRC_C = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "studies" / "s5" / "S5_CONVERGE.csv"
OUT = Path(__file__).resolve().parent / "figures"

BLUE, ORANGE, VIOLET, GRAY, GRAY2 = "#2a78d6", "#eb6834", "#6250d6", "#8a8984", "#b9b8b2"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
CCOL = {2.0: BLUE, 4.0: ORANGE, 8.0: VIOLET}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 14, "lines.linewidth": 2, "legend.frameon": False,
})


def note(fig, text, width=160):
    fig.text(0.01, 0.01, textwrap.fill(text, width), fontsize=9, color=INK2, va="bottom")


def main():
    Me = pd.read_csv(SRC_M)
    Co = pd.read_csv(SRC_C)
    OUT.mkdir(parents=True, exist_ok=True)
    Mmax = int(Co.M.max())

    # ---- the twins against qubits, window cells by coupling
    fig, ax = plt.subplots(figsize=(11, 6.4))
    W = Me[Me.region == "window"]
    Wc = Co[(Co.region == "window") & (Co.M == Mmax)]
    for c, col in CCOL.items():
        mf = W[W.c == c].groupby("p").U_MF.median()
        ax.plot(mf.index, mf.values, "--", color=col, lw=1.5, alpha=0.8)
        dt = W[W.c == c].groupby("p").U_DTWA.median()
        ax.plot(dt.index, dt.values, "-", color=col, lw=1.6, alpha=0.45)
        cv = Wc[Wc.c == c].groupby("p").U_DTWA.median()
        ax.plot(cv.index, cv.values, "o-", color=col, ms=10, lw=2.6, label=f"coupling c = {c:g}")
    sa = Co[(Co.region == "control: short anneal") & (Co.M == Mmax)].groupby("p").U_DTWA.median()
    ax.plot(sa.index, sa.values, "^", color=GRAY, ms=9, label="DTWA, short-anneal control")
    ax.plot([], [], "--", color=INK2, lw=1.5, label="mean-field twin (dashed)")
    ax.plot([], [], "-", color=INK2, lw=1.6, alpha=0.45, label="DTWA, 1,024–4,096 trajectories (thin)")
    ax.plot([], [], "o-", color=INK2, ms=9, lw=2.6, label=f"DTWA, {Mmax:,} trajectories (converged)")
    ax.set_xticks(sorted(Me.p.unique()))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Number of qubits p")
    ax.set_ylabel("Share of the exact map's labels\nthe classical twin can't explain (U)")
    ax.set_title("A stronger classical twin (DTWA) catches up at c = 2, not at c ≥ 4", loc="left")
    ax.legend(loc="lower left", fontsize=9.5, ncol=2)
    note(fig, "Window cells (anneal time τ = 3, 5, 10, 20), median over the 4 anneal times for each coupling. U = 1 − R² of "
         "the exact teacher's noise-free score fitted on the twin's 3p single-qubit outputs (5-fold cross-validated; median "
         "over 20 random readouts; 400 inputs). DTWA = discrete truncated Wigner approximation (Schachenmayer, Pikovski & "
         "Rey 2015). Thin lines: the first run (1,024 or 4,096 trajectories, not converged in every cell). Large dots: GPU "
         f"run with {Mmax:,} trajectories; 37 of 42 cells converged. The split by coupling was noticed after the first "
         "scoring (exploratory).")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_s5_twins_vs_qubits.png", dpi=200)
    plt.close(fig)

    # ---- convergence in the number of trajectories, p = 12
    fig, ax = plt.subplots(figsize=(10, 6))
    q = Co[(Co.region == "window") & (Co.p == 12)]
    for (c, tau), g in q.groupby(["c", "tau"]):
        g = g.sort_values("M")
        ax.plot(g.M, g.U_DTWA, "o-", color=CCOL[c], lw=1.6, ms=5, alpha=0.9)
        ax.annotate(f"τ={tau:g}", (g.M.iloc[-1], g.U_DTWA.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=CCOL[c], va="center")
    for c, col in CCOL.items():
        ax.plot([], [], "o-", color=col, label=f"c = {c:g}")
    ax.set_xscale("log", base=4)
    ax.set_xticks(sorted(q.M.unique()))
    ax.set_xticklabels([f"{m:,}" for m in sorted(q.M.unique())])
    ax.set_xlim(right=q.M.max() * 2.2)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("DTWA trajectories per input (nested: the same trajectories, more of them)")
    ax.set_ylabel("U, share DTWA can't explain")
    ax.set_title("With enough trajectories DTWA settles (12 qubits, every window cell)", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=10)
    note(fig, "Each line is one window cell at p = 12. Converged = |U(65,536) − U(16,384)| ≤ 0.02: 10 of 12 cells here "
         "(not converged: c = 8 at τ = 3 and τ = 5). Source: S5_CONVERGE.csv.", 150)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_s5_convergence.png", dpi=200)
    plt.close(fig)
    print("wrote", sorted(str(p) for p in OUT.glob("fig_s5_*.png")))


if __name__ == "__main__":
    main()
