#!/usr/bin/env python3
"""
"Representation efficiency" figure, made only from the tau sweep's saved run files.

  python docs/window_curves_figure.py [tau_sweep dir]   -> docs/figures/fig_window_curves_p8{,_hero}.png

For three p = 8 cells and both teacher kinds (exact, mean-field), one dataset each (seed 0, full readout,
N = 600):
  lines  = classical feature maps of growing size D (Fourier, random Fourier features, polynomial, spline):
           test AUC, logistic regression with C tuned on inner folds, mean over the 5 outer folds
           (runs/<name>_curve.csv)
  stars  = the exact-map and mean-field students (24 features: <X>,<Y>,<Z> per qubit; settings chosen
           on inner folds), mean over the same 5 folds (runs/<name>.csv)
  dashes = the best AUC possible on these labels: the noise-free teacher score itself (TT_DATASETS.csv,
           oracle_auc)
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "tau_sweep"
OUT = Path(__file__).resolve().parent / "figures"

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2, SURF = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 13, "legend.frameon": False,
})

FAMS = [("Fourier", "Fourier features", BLUE), ("Poly", "polynomial", ORANGE),
        ("RFF", "random Fourier features", AQUA), ("Spline", "splines", YELLOW)]
CELLS = [("k4_t10", "Window, a strong case\ncoupling c = 4, anneal time τ = 10"),
         ("k4_t3", "Window, its short-anneal end\nc = 4, τ = 3"),
         ("k0.5_t1", "Control: weak coupling, short anneal\nc = 0.5, τ = 1")]
TEACH = [("Q", "exact"), ("MF", "mean-field")]


def load(kind, cell):
    nm = f"tt_{kind}_p8_{cell}_local_s0"
    R = pd.read_csv(SRC / "runs" / f"{nm}.csv")
    C = pd.read_csv(SRC / "runs" / f"{nm}_curve.csv")
    T = pd.read_csv(SRC / "TT_DATASETS.csv").set_index("name").loc[nm]
    arms = R.groupby("arm").auc.mean()
    D = R.groupby("arm").D.median()        # students pick their observables per fold
    # the figure must agree with the sweep's own summary table
    for a in ("S_Q", "S_MF", "classical_best"):
        assert abs(arms[a] - T[f"auc_{a}"]) < 1e-9, (nm, a)
    return C.groupby(["family", "D"]).auc.mean().reset_index(), arms, D, T


def panel(ax, kind, cell, legend=False):
    curves, arms, D, T = load(kind, cell)
    for fam, lab, col in FAMS:
        c = curves[curves.family == fam].sort_values("D")
        ax.plot(c.D, c.auc, "-o", color=col, lw=2, ms=4, label=lab)
    ax.axhline(T.oracle_auc, color=GRAY, lw=1.2, ls="--", label="best possible (noise-free teacher)")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.plot(D["S_Q"] / 1.1, arms["S_Q"], marker="*", ms=20, color=INK, mec=SURF, mew=1, ls="none",
            label="exact-map student", zorder=5)
    ax.plot(D["S_MF"] * 1.1, arms["S_MF"], marker="D", ms=9, color=VIOLET, mec=SURF, mew=1, ls="none",
            label="mean-field student", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([8, 32, 128, 512])
    ax.set_xticklabels(["8", "32", "128", "512"])
    ax.set_ylim(0.45, 1.0)
    if legend:
        ax.legend(loc="lower left", fontsize=9.5)


def grid():
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), sharex=True, sharey=True)
    for j, (cell, title) in enumerate(CELLS):
        axs[0, j].set_title(title)
        for i, (kind, klab) in enumerate(TEACH):
            panel(axs[i, j], kind, cell)
            if j == 0:
                axs[i, j].set_ylabel(f"Labels from the {klab} map\n\ntest AUC")
            if i == 1:
                axs[i, j].set_xlabel("number of classical features")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=7, fontsize=10)
    fig.suptitle("In the window, no classical feature map catches the map that made the labels, at any size; outside it, several do",
                 fontsize=15, x=0.01, ha="left")
    fig.text(0.01, 0.01, textwrap.fill(
        "8 qubits, one labelled dataset per panel (N = 600, seed 0, full readout). Lines: logistic regression on "
        "classical features of the 8 inputs, C tuned on inner folds; AUC is the mean over 5 outer folds. Star and "
        "diamond: the exact-map and mean-field students, which choose their observables per fold (⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit = 24 "
        "features in most folds); placed at the median feature count, nudged apart so both show. Dashed: the AUC "
        "of the noise-free teacher score, the most any student can reach. Source: tau-sweep runs/tt_*_p8_*_local_s0"
        "{,_curve}.csv and TT_DATASETS.csv. Of the 63 window datasets (exact teachers, 4–8 qubits), the τ = 10 one has a "
        "larger classical shortfall than 87% and the τ = 3 one a larger one than 16% (median shortfall 0.16 AUC).", 165), fontsize=9, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.095, 1, 0.92))
    fig.savefig(OUT / "fig_window_curves_p8.png", dpi=200)
    plt.close(fig)


def hero():
    fig, ax = plt.subplots(figsize=(10, 6))
    panel(ax, "Q", "k4_t10")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.89), fontsize=9.5)
    ax.set_xlabel("number of classical features")
    ax.set_ylabel("test AUC")
    ax.set_title("In the window, the exact map does with 24 features what no classical map does with up to 1,024",
                 loc="left", fontsize=13)
    _, arms, _, T = load("Q", "k4_t10")
    ax.annotate(f"exact map, 24 features: {arms['S_Q']:.2f}", (24 / 1.1, arms["S_Q"]), xytext=(14, -4),
                textcoords="offset points", fontsize=10, color=INK, va="top")
    fig.text(0.01, 0.01, textwrap.fill(
        "8 qubits, coupling c = 4, anneal time τ = 10, labels from the exact map (N = 600, seed 0, full readout). "
        "Lines: logistic regression on classical features of the 8 inputs, C tuned on inner folds, mean AUC over 5 "
        f"outer folds; best classical choice overall {arms['classical_best']:.2f}. Dashed: noise-free teacher "
        f"({T.oracle_auc:.2f}). A strong window case: its classical shortfall (0.35 AUC) is larger than in 87% of the "
        "63 window datasets (median 0.16). Source: tau-sweep runs/tt_Q_p8_k4_t10_local_s0{,_curve}.csv.", 150),
        fontsize=9, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "fig_window_curves_p8_hero.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid()
    hero()
    print("wrote", OUT)
