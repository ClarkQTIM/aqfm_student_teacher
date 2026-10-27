# S1: How does the classical cost of matching the map grow with the number of qubits?

*Plan. **Part A: FINAL, 2026-10-27**, before any S1 teacher or fit has been computed. **Part B:
design below; its predictions are fixed in an amendment before Part B's code runs.** S1 reuses
S2's simulated banks (same input pool), but no S1 quantity has been looked at.*

## 1. The question

In the window (exact teachers, full readout, coupling c ≥ 2, τ = 3–20), the map always uses 3p
measured numbers. How much does a classical model need to match it, and how does that grow
from p = 2 to p = 12?
- **Part A:** how complex is the label function itself? The polynomial degree it takes.
- **Part B:** how big must a classical learner be (features, parameters), and how much data
  does it need?

**Context from S2:** in the window, mean-field doesn't catch up with the exact map up to
p = 16 (`studies/s2_meanfield/RESULTS.md`). The quantum part persists, so it's worth asking
whether it also costs classical learners more as p grows.

## 2. What we expect

> **Classical needs more as p grows, but about as much for mean-field teachers as for exact
> ones. If so, the cost belongs to the Ising-type map, not to quantum mechanics.**

The reason: at n = 600, classical opponents already recover *less* of the signal on
mean-field teachers (0.37–0.48) than on exact ones (about 0.6) (τ sweep, window). And
mean-field is a cheap classical computation.

**Competing outcomes, written down now:**
- **(Q) Quantum-specific growth:** the cost grows faster for exact teachers than for
  mean-field teachers.
- **(D) Small-data effect** (Part B only): with enough data, classical models close most of
  the gap at every p.
- **(F) No growth:** the label functions don't get more complex with p.

## 3. Three things that could fake a "classical cost explodes" result

1. **More inputs make almost anything harder.** We compare against matched controls at the
   same p:
   - control cells outside the window: (c = 0.5, τ = 5) and (c = 4, τ = 1);
   - J = 0 teachers.
2. **The Ising-type map itself is hard,** not quantum mechanics. So every cell has
   mean-field teachers too, and the quantum claim rests on the *difference*.
3. **Capacity vs data** (Part B). Capacity is measured at large n, then n is swept.

## 4. Data

- **Qubits:** p ∈ {2, 3, 4, 5, 6, 8, 10, 12}.
- **Cells:** S2's 14 cells: 12 window (c ∈ {2, 4, 8} × τ ∈ {3, 5, 10, 20}) and 2 controls.
- **Inputs:** the shared pool (`studies/common/bankstore.py`): x ~ N(0, I_p), clipped at ±3,
  seeded per p. That gives 40,000 rows, simulated in 100 blocks of 400.
  - Rows 0–29,999: the **training pool** (Part B draws nested n from it).
  - Rows 30,000–39,999: the fixed **test set**.
- **Teachers, at every p:**
  - exact (Q) and mean-field (MF) at each of the 14 cells, 3 readout seeds each;
  - J = 0 at each anneal time the cells use (τ ∈ {1, 3, 5, 10, 20}), 3 seeds each.

  That's 99 teachers per p.
- **The recipe** is the τ sweep's on the full readout:
  - standardize the 3p features (⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit) over the 40,000 rows;
  - draw random weights w ~ N(0, I);
  - the **noise-free score** is the standardized F·w;
  - **labels** are y ~ Bernoulli(σ(score / 0.25)).

  Weights and coin flips are seeded by the teacher's name.
- **Simulation.** Exact banks run on the GPUs: `studies/common/torchsim.py`, a line-by-line
  PyTorch port of the NumPy simulator, which a test checks against it to 10⁻¹⁰ before use.
  Mean-field and J = 0 banks run on CPU. GPU workers plus CPU workers ≤ 31.

## 5. Part A: how complex is the label function? (no learners, no label noise)

**Measurement.**
- For inputs x ~ N(0, I), polynomials of degree ≤ d are the natural complexity scale. We use
  the orthonormal Hermite basis, products of He_k(x_j)/√k!, which is well-conditioned.
- For every teacher and every degree d = 1 … d_max(p), fit its **noise-free score** by least
  squares on the training pool, and measure test R² on the 10,000 test rows.
- **V_d = 1 − R²_d** is the share of the label function that polynomials of degree ≤ d can't
  express.

**Degree limit: at most 3,000 terms**, i.e. d_max(p) is the largest d ≤ 12 with
C(p+d, d) − 1 ≤ 3,000. That's degree 4 at p = 10–12, 5 at p = 8, 7 at p = 6, 9 at p = 5, and 12 (the cap) at
p ≤ 4.
- **Why 3,000.** An out-of-sample least-squares fit with m terms on n = 30,000 rows overstates
  the unexplained part by roughly a factor 1/(1 − m/n). With 3,000 terms that's at most about
  11%. At degree 3, the primary test, it's at most 1.5%, since p = 12 has 454 terms.
- **Degree-ordered columns.** One Gram matrix per p serves every degree.

**Units for the predictions.** One unit is a teacher (kind, cell, seed).
- Its slope is the least-squares slope of V_d against p over p = 4, 5, 6, 8, 10, 12, in units
  of V per qubit.
- A group's statistic is the median slope over its units. 95% intervals come from 2,000
  bootstrap resamples of the units within each group; group differences resample each group
  independently.
- **Groups:**
  - exact in the window: 36 units;
  - exact in the controls: 6;
  - mean-field in the window: 36;
  - J = 0 at the window's anneal times: 12.

## 6. Part A predictions (fixed now)

**The test is at degree d = 3** (all cubic terms). **d = 4 is reported** with the same rules, as
a robustness check, not a test.

| | Prediction |
|---|---|
| **A1** | **Growth:** the median slope of V₃ against p, for exact teachers in the window, is > 0 (interval above 0) |
| **A2** | **Beyond input dimension:** that slope exceeds the exact-control slope and the J = 0 slope (both difference intervals above 0) |
| **A3** | **Not quantum-specific (our expectation):** exact-window minus mean-field-window slope has an interval **containing 0**. Entirely above 0 → outcome (Q); entirely below 0 → mean-field labels get harder faster |
| **A4** | *Reported, no prediction:* V_d curves for every d, and the degree needed for R² ≥ 0.90 and ≥ 0.99 (censored at d_max(p)) with its term count C(p+d, d) − 1 |

- **If A1 fails:** outcome (F) for Part A. The label functions don't need higher-degree terms as
  p grows, and Part B's growth, if any, would be about data or the learners, not the function.
- **If A1 and A2 hold but A3 doesn't (interval above 0):** that's the strong result.

## 7. Part B: how big must a classical learner be? (design; predictions fixed before it runs)

- **Reference:** the exact open-book student, logistic regression on the teacher's own 3p
  features, at the same n. The ceiling is the oracle AUC from the noise-free score.
- **"Gets there":** test AUC within 0.01 of the reference at the same n. We also report the
  recovery (AUC − 0.5)/(AUC_ref − 0.5).

| Family | Size | Grid |
|---|---|---|
| Polynomial (Hermite) + logistic regression | D | degree 1, 2, … while D ≤ 3,000 |
| Fourier + logistic regression | D | p·{1, 4, 16, 64, 256} |
| Random Fourier + logistic regression | D | same |
| Neural network (GPU) | parameters | width {16, 64, 256, 1024} × depth {2, 3, 4} |
| Gradient boosting | rounds × leaves | rounds {100, 300, 1000, 3000} × leaves {8, 31, 127} |

**Staircase.**
- **Step 1:** at n = 30,000, find size\*(p), the smallest size that gets there.
- **Step 2:** at each family's best size, sweep n ∈ {300, 1,000, 3,000, 10,000, 30,000} and find
  n\*(p).

**Draft predictions** (C1 growth, C2 beyond the controls and J = 0, C3 not quantum-specific,
C4 data) mirror A1–A3 and get fixed, with Part A's result in hand, before Part B runs.

## 8. Run order and cost

1. **`banks`** (p = 2–12, 100 blocks): exact on 2 GPUs, mean-field and J = 0 on 29 CPU
   workers.
   - The GPU time is dominated by p = 12: about 1,400 blocks of 400 rows. I'd guess about an
     hour in total; the first lines of the log will tell.
   - Blocks 0–1 already exist from S2.
2. **`teachers`**: minutes.
3. **`partA`**: least squares with at most 3,000 terms on 30,000 rows per p. Minutes.
4. **`scoreA`**: seconds.
