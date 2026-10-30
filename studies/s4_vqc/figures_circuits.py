"""Circuit diagrams (3 qubits) for the S4 students: (a) generic, (b) Hamiltonian-informed.
   python studies/s4_vqc/figures_circuits.py  -> studies/s4_vqc/figures/fig_vqc_circuits.png"""
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2, SURF = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "figure.facecolor": SURF, "savefig.facecolor": SURF})

W, H = 0.62, 0.46          # gate box size


def wires(ax, n, x0, x1, labels):
    for q in range(n):
        ax.plot([x0, x1], [-q, -q], color=INK2, lw=1.2, zorder=1)
        ax.text(x0 - 0.15, -q, labels[q], ha="right", va="center", fontsize=12)


def gate(ax, x, q, text, col, w=W, fs=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, -q - H / 2), w, H, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=col, ec="none", alpha=0.92, zorder=3))
    ax.text(x, -q, text, ha="center", va="center", color="white", fontsize=fs, zorder=4)


def cnot(ax, x, c, t):
    ax.plot([x, x], [-c, -t], color=INK, lw=1.4, zorder=2)
    ax.plot(x, -c, "o", color=INK, ms=6, zorder=4)
    ax.add_patch(plt.Circle((x, -t), 0.17, fc=SURF, ec=INK, lw=1.4, zorder=4))
    ax.plot([x - 0.17, x + 0.17], [-t, -t], color=INK, lw=1.4, zorder=5)
    ax.plot([x, x], [-t - 0.17, -t + 0.17], color=INK, lw=1.4, zorder=5)


def rzz(ax, x, a, b, text, col):
    """two-qubit ZZ rotation drawn as a box joining qubits a and b"""
    ax.plot([x, x], [-a, -b], color=col, lw=3, zorder=2)
    for q in (a, b):
        gate(ax, x, q, "", col, w=0.42)
    ax.text(x, -(a + b) / 2, text, ha="center", va="center", fontsize=8.5, color=col, zorder=6,
            bbox=dict(fc=SURF, ec="none", pad=0.5))


def bracket(ax, x0, x1, y, text, col=INK2):
    ax.plot([x0, x0, x1, x1], [y - 0.12, y, y, y - 0.12], color=col, lw=1)
    ax.text((x0 + x1) / 2, y + 0.08, text, ha="center", va="bottom", fontsize=10, color=col)


fig, axs = plt.subplots(2, 1, figsize=(15, 9.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
lab = ["|0⟩", "|0⟩", "|0⟩"]

# ---------------- generic
ax = axs[0]
wires(ax, 3, 0, 9.0, lab)
for q in range(3):
    gate(ax, 0.8, q, f"RY(x{q + 1})", GRAY, w=0.8)
ax.text(0.8, -2.55, "encode\ninputs", ha="center", va="top", fontsize=10, color=INK2)
for q in range(3):
    gate(ax, 2.2, q, f"RY(θ{q + 1})", BLUE, w=0.75)
    gate(ax, 3.2, q, f"RZ(θ{q + 4})", BLUE, w=0.75)
cnot(ax, 4.1, 0, 1); cnot(ax, 4.7, 1, 2); cnot(ax, 5.3, 2, 0)
ax.add_patch(Rectangle((1.7, -2.62), 4.0, 3.0, fill=False, ec=BLUE, ls="--", lw=1.2))
ax.text(3.7, 0.55, "layer ℓ: own angle for every rotation (2p per layer); CNOTs link neighbours", ha="center",
        va="bottom", fontsize=10, color=BLUE)
ax.text(6.55, 0.55, "× L layers (optionally re-encode the inputs each layer)", ha="left", va="bottom", fontsize=10,
        color=INK2)
for q in range(3):
    gate(ax, 8.1, q, f"read ⟨X⟩,⟨Y⟩,⟨Z⟩", INK2, w=1.5, fs=8.5)
ax.set_title("(a) Generic VQC: general-purpose layers, nothing about our map built in", loc="left", fontsize=13)

# ---------------- informed
ax = axs[1]
wires(ax, 3, 0, 9.0, lab)
for q in range(3):
    gate(ax, 0.7, q, "H", GRAY, w=0.45)
ax.text(0.7, -2.55, "start\nin |+⟩", ha="center", va="top", fontsize=10, color=INK2)
x = 1.7
for q in range(3):
    gate(ax, x, q, f"RZ(−2γ x{q + 1})", ORANGE, w=1.15, fs=9)
bracket(ax, x - 0.6, x + 0.6, 0.55, "data fields: −Σ xᵢ Zᵢ", ORANGE)
xs = [3.15, 4.05, 4.95]
for xx, (a, b) in zip(xs, [(0, 1), (0, 2), (1, 2)]):
    rzz(ax, xx, a, b, "ZZ(−2γJ)", VIOLET)
bracket(ax, 2.8, 5.3, 0.55, "coupling, every pair: −J Σ ZᵢZⱼ", VIOLET)
for q in range(3):
    gate(ax, 6.2, q, "RX(−2β)", AQUA, w=0.95)
bracket(ax, 5.7, 6.7, 0.55, "driver: −Σ Xᵢ", AQUA)
ax.add_patch(Rectangle((1.05, -2.62), 5.85, 3.0, fill=False, ec=INK, ls="--", lw=1.2))
ax.text(4.0, 0.95, "layer ℓ: only two trainable numbers, γ_ℓ (data + coupling time) and β_ℓ (driver time)",
        ha="center", va="bottom", fontsize=10.5)
ax.text(7.0, 0.55, "× L layers\n(new γ, β each)", ha="left", va="bottom", fontsize=10, color=INK2)
for q in range(3):
    gate(ax, 8.1, q, "read ⟨X⟩,⟨Y⟩,⟨Z⟩", INK2, w=1.5, fs=8.5)

# inset: ZZ rotation = CNOT, RZ, CNOT
x0 = 10.4
ax.plot([x0, 14.4], [-0.4, -0.4], color=INK2, lw=1); ax.plot([x0, 14.4], [-1.6, -1.6], color=INK2, lw=1)
rzz(ax, x0 + 0.5, 0.4, 1.6, "ZZ(φ)", VIOLET)
ax.text(x0 + 1.15, -1.0, "=", fontsize=16, ha="center", va="center")
cnot(ax, x0 + 1.8, 0.4, 1.6)
gate(ax, x0 + 2.6, 1.6, "RZ(φ)", VIOLET, w=0.75)
cnot(ax, x0 + 3.4, 0.4, 1.6)
ax.text(x0 + 1.9, -2.35, "each pair's coupling in standard gates", ha="center", fontsize=9.5, color=INK2)
ax.set_title("(b) Hamiltonian-informed VQC (QAOA-style): each layer is a short burst of our map's own three pieces",
             loc="left", fontsize=13)

for ax in axs:
    ax.set_xlim(-0.9, 15)
    ax.set_ylim(-2.9, 1.5)
    ax.axis("off")
fig.text(0.01, 0.01, textwrap.fill(
         "3 qubits shown. RZ(θ) = exp(−iθZ/2), RX likewise; ZZ(φ) = exp(−iφ ZᵢZⱼ/2). The data and coupling parts are both "
         "diagonal, so they can be applied in any order exactly. At p qubits the coupling needs p(p−1)/2 ZZ gates per layer "
         "(3 here, 28 at p = 8). The real anneal is this same pattern repeated ~1,600 times with tiny, fixed γ and β that "
         "follow the schedule; the informed VQC uses a few layers with γ, β learned.", 190),
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(str(Path(__file__).resolve().parent / "figures" / "fig_vqc_circuits.png"), dpi=170)
print("ok")
