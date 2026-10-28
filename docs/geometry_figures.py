#!/usr/bin/env python3
"""
Figure for docs/geometry_results.md, made only from runs/tau_sweep/geometry/GEO_TT.csv.

  python docs/geometry_figures.py [GEO_TT.csv]    -> docs/figures/fig_geometry_vs_gap.png

Each dot is one tau-sweep dataset (exact teacher, full readout, p = 4-8):
  x = geometric difference g(mean-field || exact) at lambda = 0.01, best Gaussian width (label-free)
  y = measured AUC gap, exact student minus mean-field student (from the tau sweep's runs)
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "tau_sweep" / "geometry" / "GEO_TT.csv"
OUT = Path(__file__).resolve().parent / "figures"

BLUE, GRAY2 = "#2a78d6", "#b9b8b2"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 12, "axes.titlesize": 14, "legend.frameon": False,
})


def main():
    G = pd.read_csv(SRC)
    q = G[(G.kind == "Q") & (G.readout == "local")]
    win, other = q[q.region == "window"], q[q.region != "window"]
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(other.g_MF_rbf, other.gap_Q_MF, s=40, color=GRAY2, edgecolor=SURF, lw=1.5,
               label=f"outside the window ({len(other)} datasets)")
    ax.scatter(win.g_MF_rbf, win.gap_Q_MF, s=40, color=BLUE, edgecolor=SURF, lw=1.5,
               label=f"window ({len(win)} datasets)")
    ax.axhline(0.05, color=INK2, lw=1, ls=":")
    ax.annotate("0.05 AUC", (ax.get_xlim()[1], 0.05), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=INK2, fontsize=9)
    ax.set_xlabel("How differently the exact map and mean-field arrange the inputs\n"
                  "geometric difference g(mean-field ‖ exact), no labels used")
    ax.set_ylabel("How much the exact student beat the mean-field one\n(AUC gap, measured)")
    ax.set_title("A label-free measure from theory picks out the same window")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, textwrap.fill(
        "Each dot is one τ-sweep dataset: exact teacher, full readout, 4–8 qubits. g is Huang et al. (2021)'s geometric "
        "difference (regularized, λ = 0.01). Pre-registered check G2: AUROC 0.95 for 'exact − mean-field ≥ 0.05', "
        "Spearman 0.87. Source: GEO_TT.csv.", 150), fontsize=9, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_geometry_vs_gap.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "fig_geometry_vs_gap.png")


if __name__ == "__main__":
    main()
