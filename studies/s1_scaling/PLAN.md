# S1: How does the classical cost of matching the map grow with the number of qubits?

*Plan, DRAFT 2026-10-26. Nothing has been run. Runs after S2.*

## 1. The question

In the window (exact teachers, full readout, c ≥ 2, τ = 3–20), the map always uses 3p
measured numbers. How much does a classical model need to match it, and how does that grow
from p = 2 to p = 12? The classical side is measured three ways:

- **features:** polynomial, Fourier, random Fourier;
- **parameters:** neural networks, boosted trees;
- **training samples.**

## 2. What we expect

> **Classical needs more as p grows, but about as much for mean-field teachers as for exact
> ones. If so, the cost belongs to the Ising-type map, not to quantum mechanics.**

The reason: at n = 600, classical opponents already recover *less* of the signal on
mean-field teachers (0.37–0.48) than on exact ones (about 0.6) (τ sweep, window). And
mean-field is a cheap classical computation.

**Competing outcomes, written down now:**
- **(Q) Quantum-specific growth:** the cost for exact teachers grows *faster* than for
  mean-field teachers. That's the strong result.
- **(D) Small-data effect:** with enough data (n = 30,000), classical models close most of the
  gap at every p.

## 3. Three things that could fake a "classical cost explodes" result

1. **More inputs make almost anything harder.** We compare against matched controls at the
   same p:
   - cells outside the window: (c = 0.5, τ = 5) and (c = 4, τ = 1);
   - J = 0 teachers, whose features each depend on one input.
2. **The Ising-type map itself is hard,** not quantum mechanics. So every cell has
   **mean-field teachers** too, and the quantum claim rests on the *difference* between exact
   and mean-field curves.
3. **Capacity vs data.** At small n, bigger models overfit. So we measure capacity at a large
   n, then sweep n.

## 4. Datasets

- **p** ∈ {2, 3, 4, 5, 6, 8, 10, 12}.
- **Cells:** the same 14 as S2: 12 window cells (c ∈ {2, 4, 8} × τ ∈ {3, 5, 10, 20}) plus the
  2 controls.
- **Teachers:** exact, mean-field and J = 0, with 3 seeds each. Seeds differ only in the random
  readout weights, so they share one simulation.
- **Labels:** the τ-sweep recipe: random weights on the standardized full readout, then a coin
  flip at T = 0.25.
- **Samples:** the shared pool from `studies/common`. Rows 0–29,999 are the training pool,
  rows 30,000–39,999 the fixed **test set**.
  - n ∈ {300, 1,000, 3,000, 10,000, 30,000}, as nested subsets of the training pool.
  - Tuning uses a validation split inside the training rows, never the test set.

## 5. Part A: how complex is the label function? (no learners, no label noise)

Fit each teacher's **noise-free score** with polynomials of degree d = 1, 2, … by least
squares, and measure R² on the test set.
- We go as far as the number of terms, C(p+d, d) − 1, allows: at most 20,000, which is
  degree 6 at p = 12.
- For Gaussian inputs, polynomials (the Hermite basis) are the natural yardstick of
  complexity.
- **d₉₀(p), d₉₉(p):** the degree needed for R² ≥ 0.90 and 0.99, and the matching feature
  count.

## 6. Part B: how big must a classical learner be? (noisy labels, AUC)

**Reference.** The exact "open book" student: logistic regression on the teacher's own 3p
features, trained on the same n. The oracle AUC, from the noise-free score, is the ceiling.

**Gets there** means test AUC within 0.01 of the reference at the same n. That's the τ sweep's
D* rule. We also report the recovery (AUC − 0.5)/(AUC_ref − 0.5).

| Family | Size | Grid |
|---|---|---|
| Polynomial + LR | D = C(p+d, d) − 1 | degree 1, 2, … while D ≤ 20,000 |
| Fourier + LR | D | p·{1, 4, 16, 64, 256, 1024} |
| Random Fourier + LR | D | same |
| Neural network | parameters | width {16, 64, 256, 1024} × depth {2, 3, 4} |
| Gradient boosting | rounds × leaves | rounds {100, 300, 1000, 3000} × leaves {8, 31, 127} |
| RBF SVM / kernel ridge | n only | — |

**Staircase.**
- **Step 1:** at n = 30,000, find **size\*(p)**, the smallest size that gets there, or "not
  within the grid".
- **Step 2:** at each family's best size, sweep n and find **n\*(p)**.

## 7. Predictions (fixed when this plan is committed)

Each slope is fitted on the window cells against p, with a 95% bootstrap interval over cells
and seeds.

| | Prediction |
|---|---|
| **C1** | **Growth:** for exact teachers, log d₉₉ and log size\* of the best classical family grow with p (interval above 0) |
| **C2** | **Beyond input dimension:** that growth is faster in the window than in the control cells and for J = 0 teachers |
| **C3** | **Not quantum-specific (our expectation):** the exact-teacher slope minus the mean-field-teacher slope has an interval containing 0. If it's entirely above 0 → outcome (Q) |
| **C4** | **Data:** n\*(p) for the best classical family grows with p in the window, faster than in the controls. If classical gets there at every p by n = 30,000 → outcome (D) |

## 8. Cost and open decisions

- **Simulation:** at p = 12, 14 cells × 40,000 samples is about 5 hours on 31 cores, or much
  less on GPU (`AQFM_GPU=1`). p = 10 takes about 1 hour; everything smaller takes minutes.
- **Learners:** the big neural networks on 30,000 samples are the expensive part.
  - **Open:** can they use the L40 / H100? If not, the network grid gets trimmed.
- **Open:** keep all 12 window cells, or cut to 6 to halve the p = 12 time?
