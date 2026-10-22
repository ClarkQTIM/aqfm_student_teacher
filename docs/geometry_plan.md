# Geometric difference: plan

*Written 2026-10-22, before the analysis has been run on any of our data. The only numbers
looked at so far come from random matrices in the unit tests. As with the other plans, the
predictions below are fixed. Changes go at the end as dated amendments.*

Code: `experiments/ts_geometry.py`. Tests: `tests/test_geometry.py`. No new quantum
simulation: everything is built from banks that already exist on the server.

## 1. Why

Huang et al. (2021), *Power of data in quantum machine learning*, Nat. Commun. 12, 2631
([arXiv:2011.01938](https://arxiv.org/abs/2011.01938)), is the paper a reviewer will hold our
teacher–student work against.

- **What they do.** They generate labels from a quantum model and ask when classical learners
  can keep up, which is our design.
- **Their test.** They give a label-free quantity, the **geometric difference** g between a
  classical kernel and a quantum one.
  - If g is small, no labelling of those samples can favour the quantum kernel by much.
  - If g is large, an advantage is possible, and the labels decide whether it happens.
- **What we do here.** Compute g on our own data, so our empirical findings can be read in
  their terms:
  - the window (strong coupling, τ = 3–20, full readout);
  - "mean-field labels are classically hard too";
  - the real-data nulls.

What it adds:
- **A theory-backed check of the window.** If g is large in the window and small outside it,
  the window isn't an artefact of our particular opponents. If g is small in the window, the
  window says more about our menus than about the map, and we need to know that.
- **The twin, in Huang's language.** We also compute g of the map's own mean-field version
  against the exact map. As far as we know, nobody has done this; to be checked in a proper
  literature search.
- **A label-free rule-out for real data.** g needs no labels, so it can rule a dataset out
  before any model is fit. That's the screen's first step, now with theory behind it.
- **A number for the sample-size sweep.** g² is roughly how many times more data the emulator
  needs (Section 2), so it predicts where in n classical should catch up.

## 2. Definitions

**Huang's Eq. 5.** For two kernel matrices on the same N samples, both scaled to trace N:

    g(K_C ‖ K_T) = sqrt( ‖ √K_T  K_C⁻¹  √K_T ‖ )        (‖·‖ = largest eigenvalue)

Here T is the target (the model whose labels we care about) and C is the emulator (who tries
to copy them). Their prediction-error bound is E|h − f| ≤ c·sqrt(s/N). Here s is the model
complexity (|w|² of the fitted model). For labels the target can express, s_C ≤ g² s_T. So
the emulator needs about **g² times more samples** to reach the same bound.

**Why we need a regularized version.** Eq. 5 inverts K_C. Our Gaussian kernels are close to
singular, and the linear kernels on map features are exactly singular (rank ≤ 108 out of
N = 600). With kernel ridge regression at ridge λ, the same steps give the version below. Huang
et al. have a regularized version in their supplement, which we couldn't open from here. We
derived ours from the same argument; to be compared with theirs.

Take any labels the target can express: y = √K_T v, with s_T = |v|². Kernel ridge with the
emulator gives α = (K_C + λ)⁻¹ y. Then:

- **Complexity:** s_C = |w|² = yᵀ (K_C+λ)⁻¹ K_C (K_C+λ)⁻¹ y = vᵀ √K_T √K_C (K_C+λ)⁻² √K_C √K_T v,
  so **s_C ≤ g_λ² s_T** with

      g_λ² = ‖ √K_T √K_C (K_C+λ)⁻² √K_C √K_T ‖

- **Training residual:** y − K_C α = λ (K_C+λ)⁻¹ y, so the mean squared training error is
  **≤ t_λ s_T** with

      t_λ = ‖ √K_T λ² (K_C+λ)⁻² √K_T ‖ / N

As λ → 0 with K_C invertible, g_λ → Eq. 5 and t_λ → 0. The tests check exactly that, plus both
inequalities on random kernels. **The two numbers go together:**
- g_λ is "what it costs the emulator, in data, to fit";
- t_λ is "what it can't fit at all".

A low-rank emulator can show a small g_λ only because it gives up on most directions (large
t_λ). That's why the main tests below use full-rank (Gaussian) emulators.

**λ** is in units of the mean eigenvalue: trace N means the eigenvalues average 1. The grid is
λ ∈ {0.001, 0.01, 0.1, 1}. **The primary value is λ = 0.01**, fixed now. The other three are
reported as sensitivity checks; they aren't tests.

**Label-aware quantity** (Huang's second step, descriptive only): s_λ(y; K) and the training
error for the actual labels, y = ±1.

## 3. Kernels

**Target (T): what the map's linear readout can express.**
- **τ sweep:** the linear kernel on the teacher's own standardized readout (⟨X⟩,⟨Y⟩,⟨Z⟩ per
  qubit, or ⟨Z⟩ only), from the teacher's own simulator, coupling and τ. Before the coin flip,
  every teacher label is exactly a linear function of these features. So K_T describes the
  label family exactly, and it's what our map students use.
- **Real data:** the linear kernel on the exact map's **full** readout (3p + 3q features) at
  each of the 60 (coupling, τ) settings, plus the exact student's own discovery choice (family,
  coupling, τ).

**Emulators (C).** We always credit the emulator with its best width: g is the minimum over
three Gaussian widths, γ = c / median squared distance, c ∈ {0.25, 1, 4}.
- **classical:** Gaussian kernel on the inputs z. Huang et al. use Gaussian kernels among their
  classical comparisons.
- **MF_rbf / J0_rbf / Q_rbf (twins):** Gaussian kernel on the twin's features, with the same
  readout and the same coupling and τ (J=0: the same τ). This is *stronger* than our linear
  twin students, so a large g here is a conservative result.
- **MF_lin / J0_lin / Q_lin:** linear kernel on the twin's features, i.e. what the linear twin
  student can express. It's low-rank, so it's reported with its t_λ and not used in the tests.

**N.** The τ sweep uses its 600 rows. For real data, datasets with more than 600 rows are
subsampled to 600 (seeded by name), because g grows with N (g² ≤ N / 4λ). Smaller datasets use
all their rows and are flagged; their g is not directly comparable. Real inputs are also often
discrete, unlike the τ sweep's Gaussian inputs, so g₀ (G7) is a calibrated guess, not a law.

**What g doesn't cover.** g compares kernel methods. It says nothing directly about gradient
boosting or the neural net, so it complements the ladder; it doesn't replace it.

## 4. Outcomes we compare against (already computed, not recomputed)

From `runs/tau_sweep/ADDONS_DATASETS.csv`:
- **classical shortfall:** cls_gap = best map student − max(classical_best, nonlinear_best);
- **twin gap:** gap_Q_MF = S_Q − S_MF;
- **region:** window, weak-or-short, κ=1, other, Z-only.

From `REAL_DATASETS.csv`: verdict, confirmation margin interval, compactness.

## 5. Predictions

All at λ = 0.01, exact teachers, full (local) readout unless stated. The "gap" threshold is
0.05 AUC, as everywhere else.

| | Prediction | Why |
|---|---|---|
| **G1** | g_classical predicts "classical short by ≥ 0.05": **AUROC ≥ 0.75**, and **Spearman(g, shortfall) ≥ 0.5** | If the window is real, the map's kernel should be geometrically far from the best Gaussian kernel there |
| **G2** | g(MF ‖ Q) predicts "exact − mean-field ≥ 0.05": **AUROC ≥ 0.75**, **Spearman ≥ 0.5** | The same, for the twin |
| **G3** | Median g(MF ‖ Q) in the window is **≥ 2×** the median in the weak-coupling / short-anneal region | Mean-field is accurate there; the window is where it breaks |
| **G4** | In window cells (κ ≥ 2, τ 3–20, all p), g_classical with **Z-only** readout is below g_classical with full readout in **≥ 80%** of cells | Z-only labels were easy to copy everywhere (median margin 0; none of 180 reached 0.05) |
| **G5** | Mean-field teachers in the window: median g_classical is **≥ 0.5×** that of exact teachers | Mean-field labels were just as hard for classical opponents (best opponent about 0.71, vs 0.78 on exact labels). This is that finding in Huang's terms |
| **G6** | *Descriptive:* g² in the window, per p | The predicted extra-data factor; it picks the n-sweep cells and their predicted catch-up n |
| **G7** | **Rule-out.** Threshold g₀ = the 5th percentile of g_classical over τ-sweep datasets (all kinds, both readouts) that classical opponents miss by ≥ 0.05. Real datasets whose largest g_classical over the 60 settings is below g₀: **none** has a verdict other than "classical enough" or a confirmation interval above 0 | A small g means no labels on those inputs can favour the map much |
| **G8** | Among the pool datasets, **monk2** has one of the **three largest** ratios s(MF_rbf) / s(target) for its actual labels, at the exact student's choice | monk2 is the one real dataset with a confirmed exact-over-mean-field gap |
| **G9** | *Descriptive:* the share of real datasets below g₀ | |

**The bar for G1 and G2.** Both parts must hold at λ = 0.01. If they hold at 0.01 but not at
the other λ values, we say so.

**If G1 or G2 fails,** the window is not where the geometry says the map differs from Gaussian
kernels, or from its mean-field twin. That would mean our empirical window comes from finite
samples or from our menus, and the paper's framing changes. It's the most important outcome to
watch.

## 6. What happens next with the results

- **n sweep.** Choose cells across the range of g (window and outside). Predict the catch-up n
  from g² × 600, and test it.
- **Screen and real data.** If G7 holds, g becomes the screen's step 0: label-free, computed
  before any model is fit.
- **Write-up.** g goes into `aqfm_overview.tex` as the bridge to Huang et al.
