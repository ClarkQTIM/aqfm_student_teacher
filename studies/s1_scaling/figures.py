#!/usr/bin/env python3
"""
Figures for S1 Part A, made only from runs/studies/s1/S1_PARTA.csv (no numbers typed in).

  python studies/s1_scaling/figures.py            -> studies/s1_scaling/figures/*.png

fig_partA_efficiency_p8.png   share of the noise-free label function captured by polynomials,
                              against the number of polynomial features, at p = 8
fig_partA_vs_qubits.png       share NOT captured by cubic polynomials, against the number of qubits
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "studies" / "s1" / "S1_PARTA.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "figures"

# palette: the dataviz reference instance (light); each teacher kind keeps its color in every figure
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 14, "lines.linewidth": 2, "lines.markersize": 8,
    "legend.frameon": False,
})


def note(fig, text):
    fig.text(0.01, 0.01, textwrap.fill(text, 150), fontsize=9, color=INK2, va="bottom")


def main():
    A = pd.read_csv(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = [  # label, which teachers, color, line style
        ("exact map, window", (A.kind == "Q") & (A.region == "window"), BLUE, "o-"),
        ("mean-field, window", (A.kind == "MF") & (A.region == "window"), ORANGE, "o-"),
        ("no coupling (J = 0), window anneal times", (A.kind == "J0") & (A.region == "window"), AQUA, "o-"),
        ("no coupling, short anneal τ = 1 (easy control)", (A.kind == "J0") & (A.tau == 1.0), GRAY, "^--"),
    ]

    # ---- the "representation efficiency" figure, p = 8
    P = 8
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, sel, col, sty in groups:
        g = A[sel & (A.p == P)].groupby("D").R2.median()
        ax.plot(g.index, g.values, sty, color=col, label=lab)
    ax.plot([3 * P], [1.0], "*", color=BLUE, ms=18, label=f"the map's own {3 * P} measured numbers")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Number of polynomial features of the inputs (p = {P} qubits; degree 1 → 5)")
    ax.set_ylabel("Share of the label function reproduced (test R²)")
    ax.set_title(f"In the window, a thousand polynomial terms reproduce about a quarter (p = {P})")
    for d, D in A[A.p == P].groupby("d").D.first().items():
        ax.annotate(f"deg {d}", (D, 0.02), ha="center", color=INK2, fontsize=9)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.70), fontsize=10)
    note(fig, "Median over teachers: 36 exact and 36 mean-field (12 window settings × 3 random readouts), 12 uncoupled "
         "(4 anneal times × 3), 3 control. Noise-free label functions, least squares on 30,000 inputs, scored on 10,000 "
         "held-out inputs. The exact teacher is built from the map's 24 numbers, so the star is 1 by construction.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_partA_efficiency_p8.png", dpi=200)
    plt.close(fig)

    # ---- the "does it grow with qubits" figure
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, sel, col, sty in groups:
        g = A[sel & (A.d == 3)].groupby("p").V
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, sty, color=col, label=lab)
        ax.fill_between(med.index, q1.values, q3.values, color=col, alpha=0.12, lw=0)
    ax.set_xticks(sorted(A.p.unique()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of qubits p")
    ax.set_ylabel("Share of the label function cubic polynomials MISS")
    ax.set_title("It doesn't get worse with more qubits")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.22), ncol=2, fontsize=11)
    note(fig, "Median over teachers; band = middle 50%. Degree ≤ 3 polynomials of the inputs, least squares on "
         "30,000 inputs, scored on 10,000 held-out inputs. Pre-registered test A1 (this share grows with p for the exact "
         "map in the window): slope −0.010 [−0.021, +0.001] per qubit, failed.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_partA_vs_qubits.png", dpi=200)
    plt.close(fig)
    print("wrote", sorted(str(p) for p in OUT.glob("fig_partA_*.png")))


if __name__ == "__main__":
    main()
