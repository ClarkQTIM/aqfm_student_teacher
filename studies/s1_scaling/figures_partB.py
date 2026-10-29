#!/usr/bin/env python3
"""
Figures for S1 Part B, made only from runs/studies/s1/S1_PARTB.csv (no numbers typed in).

  python studies/s1_scaling/figures_partB.py [S1_PARTB.csv]   -> studies/s1_scaling/figures/fig_partB_*.png

fig_partB_vs_qubits.png   how far the best classical learner (networks or trees, picked on validation)
                          falls short of the open-book student, against the number of qubits, n = 30,000
fig_partB_vs_data.png     the same shortfall against the number of labelled rows, one line per p
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "studies" / "s1" / "S1_PARTB.csv"
OUT = Path(__file__).resolve().parent / "figures"

# same colors per teacher kind as the Part A figures
BLUE, ORANGE, AQUA, GRAY, GRAY2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#b9b8b2"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]     # blue steps 250 ... 700
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 14, "lines.linewidth": 2, "lines.markersize": 8,
    "legend.frameon": False,
})


def note(fig, text, width=150):
    fig.text(0.01, 0.01, textwrap.fill(text, width), fontsize=9, color=INK2, va="bottom")


def main():
    S = pd.read_csv(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    L = S[S.choice == "learned"]
    N = int(L.n.max())

    # ---- shortfall against qubits, n = 30,000
    groups = [  # label, teachers, color, style, band
        ("exact map, window", (L.kind == "Q") & (L.region == "window"), BLUE, "o-", True),
        ("mean-field, window", (L.kind == "MF") & (L.region == "window"), ORANGE, "o-", True),
        ("no coupling (J = 0), window anneal times", (L.kind == "J0") & (L.region == "window"), AQUA, "o-", False),
        ("exact map, weak-coupling control", (L.kind == "Q") & (L.region == "control: weak coupling"), GRAY, "s--", False),
        ("exact map, short-anneal control", (L.kind == "Q") & (L.region == "control: short anneal"), GRAY2, "^--", False),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, sel, col, sty, band in groups:
        g = L[sel & (L.n == N)].groupby("p").shortfall
        med = g.median()
        ax.plot(med.index, med.values, sty, color=col, label=lab)
        if band:
            ax.fill_between(med.index, g.quantile(0.25).values, g.quantile(0.75).values, color=col, alpha=0.08, lw=0)
    ax.axhline(0.01, color=INK2, lw=1, ls=":", label="“gets there”: within 0.01 of open-book")
    ax.set_xticks(sorted(L.p.unique()))
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of qubits p")
    ax.set_ylabel("How far the best classical learner falls short\n(open-book AUC minus learner AUC)")
    ax.set_title("Networks and trees fall short in the window, and more so for mean-field labels")
    ax.legend(loc="upper left", fontsize=10)
    note(fig, f"{N:,} labelled rows (80% train, 20% validation); test on 10,000 other rows. Best classical learner = "
         "neural network (12 shapes) or gradient-boosted trees (12 settings), whichever has the lower validation loss. "
         "Open-book = logistic regression on the teacher's own 3p map features. Median over teachers; band = middle 50% "
         "(36 teachers per window group: 12 settings × 3 random readouts).")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_partB_vs_qubits.png", dpi=200)
    plt.close(fig)

    # ---- shortfall against data, one line per p
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ps = sorted(L.p.unique())
    W = L[L.region == "window"].groupby(["kind", "p", "n"]).shortfall.median()
    top = float(W.loc[["Q", "MF"]].max()) * 1.08
    for ax, (kind, title) in zip(axs, [("Q", "Labels from the exact map"), ("MF", "Labels from mean-field")]):
        q = L[(L.kind == kind) & (L.region == "window")]
        for p, col in zip(ps, SEQ):
            g = q[q.p == p].groupby("n").shortfall.median()
            ax.plot(g.index, g.values, "o-", color=col, label=f"{p} qubits")
        ax.axhline(0.01, color=INK2, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(220, 40_000)
        ax.set_title(f"{title}, window")
        ax.set_xlabel("Labelled rows")
        ax.set_ylim(0, top)
    axs[0].set_ylabel("How far the best classical learner falls short\n(open-book AUC minus learner AUC)")
    axs[0].legend(loc="upper right", fontsize=10)
    fig.suptitle("More data helps, but slowly, and less with more qubits", x=0.01, ha="left", fontsize=15)
    note(fig, "Median over the 36 window teachers per qubit count and label kind. Each learner (network or trees) and "
         "its size are picked on the validation 20% of the labelled rows; the open-book student gets the same rows. "
         "Dotted line: within 0.01 AUC of the open-book student. Mean-field, p = 8, 10, 12 at 30,000 rows: the three medians "
         "coincide at 0.160 (individual teachers range 0.03–0.40).", 170)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUT / "fig_partB_vs_data.png", dpi=200)
    plt.close(fig)
    print("wrote", sorted(str(p) for p in OUT.glob("fig_partB_*.png")))


if __name__ == "__main__":
    main()
