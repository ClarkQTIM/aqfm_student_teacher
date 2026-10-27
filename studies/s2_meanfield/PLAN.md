# S2: Does mean-field catch up with the exact map as p grows?

*Plan, FINAL 2026-10-26. The predictions (Section 5) are exactly as committed in the draft
(6bb5cf7); the changes made while writing the code are listed in Section 8.*

## 1. Why this comes first

We couple every qubit to every other with J = c/(p−1). With that normalization, each qubit
feels the *average* of many weak pulls. As p grows, mean-field (each qubit alone in the
average field of the others) is expected to become exact. If it does, the window, where the
exact map beats mean-field by 0.14–0.24 AUC at p = 4–8, is a small-system effect of our
coupling choice.

Two earlier results disagree on this:
- the geometric difference g(MF ‖ exact) in the window **fell** from 18.6 to 11.4 between
  p = 4 and 8 (`docs/geometry_results.md`);
- the measured exact-over-mean-field AUC gap **didn't** (0.19, 0.14, 0.24).

This study measures mean-field's error directly. There are no labels to learn and no
learners, so it's cheap and can go beyond p = 12.

## 2. What we expect

> **The error falls everywhere, but more slowly in the window**, because that's where the
> anneal crosses the ordering transition and fluctuations stay large for longer.

**What the outcomes mean:**
- **Falls fast in the window too:** the window is a small-system effect of all-to-all
  coupling. S3 (other couplings) becomes the priority.
- **Falls slowly in the window:** the window survives over a practically relevant range of p.
- **Flat or growing:** the strongest outcome, and not what we expect.

The expected rates (fast convergence away from the transition, slow near it) are our
reasoning from standard mean-field and finite-size behaviour. We still have to find proper
references for this model before any write-up.

## 3. Design

- **Qubits:** p ∈ {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}. p = 16 is included if the measured
  cost holds (Section 6).
- **Cells:**
  - **window:** c ∈ {2, 4, 8} × τ ∈ {3, 5, 10, 20}, 12 cells;
  - **controls:**
    - (c = 0.5, τ = 5): weak coupling;
    - (c = 4, τ = 1): short anneal, strongly entangled but easy to copy.
- **Inputs:** 400 samples per p, x ~ N(0, I_p), z = clip(x, −3, 3).
  - The samples are block 0 of the shared input pool (`studies/common`), which S1 extends
    later.
  - No standardization by the sample, so adding samples later never changes earlier ones.
    That's a small change from the τ sweep, which standardized by the sample mean and SD.
- **Simulations:** exact (Strang) and mean-field at every (p, c, τ). Both start from |+⟩ and
  use the same step rule as the τ sweep.

## 4. What we measure

For each (p, cell), over the 400 samples:

1. **Readout error, E.** The root-mean-square distance between exact and mean-field Bloch
   vectors (⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩), over samples and qubits. This is the most direct "how
   wrong is mean-field" number.
2. **Missing correlations, C.** Mean-field sets every connected correlation
   ⟨P_i P_j⟩ − ⟨P_i⟩⟨P_j⟩ to zero; we measure the exact ones.
   - **C_pair:** the typical size of one pair's correlation, the mean |·| over pairs,
     P ∈ {X, Y, Z} and samples. **C_ZZ** is the same for Z only.
   - **C_coll:** what the correlations add up to on one qubit, J · |Σ_j C^ZZ_ij| (the sum is
     signed), averaged over i and samples. Correlations that cancel don't count here.
3. **Label-level error, U.** How much of an exact teacher's noise-free score mean-field
   features *cannot* explain.
   - Use 20 random readout weight vectors per cell, the τ-sweep recipe without the coin flip.
   - For each, regress the exact score on the 3p mean-field features from the same (c, τ),
     using linear least squares with 5-fold cross-validation.
   - U = 1 − R², the median over the 20.
   
   This ties the physics back to what a student sees.

   One known bias: an out-of-fold fit with 3p features on 320 training rows overstates U by
   roughly 3p/320. That's about 4% of U at p = 4 and 15% at p = 16, which makes U look like it
   falls *more slowly* than it does. So it works against M4, not for it.

**Exponent.** For each cell and measure, fit log(measure) = a − α·log p over p ≥ 4. We get a
95% interval for α by bootstrapping the samples. Larger α means mean-field catches up faster.
p = 2–3 are reported but left out of the fit, since they're too small for a power law.

## 5. Predictions (fixed when this plan is committed)

| | Prediction |
|---|---|
| **M1** | **Controls:** the readout error E falls with p in both control cells (α's 95% interval above 0) |
| **M2** | **Window:** E falls with p in at least 9 of the 12 window cells (α's interval above 0) |
| **M3** | **Slower in the window:** the median α for E over window cells is below the median over control cells, with the bootstrap interval of the difference below 0 |
| **M4** | **Labels:** the label-level error U falls with p in the window (median α over window cells > 0, interval above 0) |
| **M5** | *Reported, no prediction:* C_pair, C_ZZ and C_coll against p, and the p at which U would drop below 0.01 if each window cell's fitted power law held. That last one is an extrapolation, labelled as such |

M2 and M3 together are "falls everywhere, but more slowly in the window". If M2 fails, mean-field
is *not* catching up in the window over this range of p, which would be the surprising outcome.

## 6. Cost (one core, measured 2026-10-26)

Exact simulation per sample: p = 12 takes 0.37 s at τ = 3 and 2.2 s at τ = 20. Each extra two
qubits costs roughly 4–5× more.

- **p ≤ 12:** 14 cells × 400 samples, minutes in total.
- **p = 14:** about 10 s per sample at τ = 20. About 15 minutes on 31 cores.
- **p = 16:** about 50 s per sample at τ = 20. About an hour on 31 cores. We time one block
  before committing to it.
- **Mean-field:** negligible at every p.

## 7. What this doesn't do

- It doesn't touch classical learners (that's S1) or other couplings (that's S3).
- The exponents are fitted over p = 4–16. They describe that range; they aren't
  infinite-p limits.

## 8. Changes from the draft, and how it runs

**Changes, made while writing the code:**
- **Correlation measure.** The draft's "C_feedback = J · Σ_j |C_ij|" is, for our uniform
  coupling, exactly c × (the mean |C_ij| over pairs), so it would have added nothing. It's replaced by C_coll (signed
  sum, Section 4), and C_ZZ is added. Both are report-only (M5).
- **Bias note on U** added to Section 4.
- **No prediction changed.**

**Disclosure.** While testing the code, a smoke run computed the measures at p = 2–6 (all 14
cells, 400 rows), after the draft predictions were committed. The predictions above are
unchanged from that commit.

**Code:** `studies/s2_meanfield/run.py`, with phases `time`, `banks`, `measure`, `score`.
Shared code is in `studies/common/`. Tests: `tests/test_studies_s2.py`.

**Outputs:**
- simulations: `runs/studies/banks/alltoall/`;
- results: `runs/studies/s2/S2_MEASURES.csv`, `S2_ALPHAS.csv`, `S2_CHECKS.csv`.

**p = 16:** runs only if `time --p 16` estimates under about 2 hours on 31 workers. Its exponents
are then refitted with p = 16 included, and both fits are reported.


## 9. Amendment 1 (2026-10-27, AFTER seeing the primary results)

**What the primary run found** (p = 2–14, `runs/s2.log`):
- **M1 held:** mean-field converges in both controls.
- **M2 failed:** the readout error E falls in only 5 of 12 window cells.
- **M3 held,** trivially.
- **M4 failed:** U in the window doesn't fall (median α = −0.09 [−0.10, −0.04]); it stays near
  0.82 from p = 2 to 14.

These results stand as reported. The amendment adds two checks, because the failure of M4 is
the headline and could have two mundane causes:

- **A1: the best mean-field setting, not the matched one.** U compares the exact teacher with
  mean-field at the *same* (c, τ). The τ sweep's mean-field student could choose any of 60
  settings (c ∈ {0.5, 1, 2, 4, 8} × the 12 τ values). So for each exact teacher, U_best uses
  the setting with the smallest out-of-fold error.
  - Picking on the same errors flatters mean-field slightly. That works *against* "mean-field
    can't catch up".
  - Every cell's own setting is in the menu, so U_best ≤ U.
- **A2: 800 rows instead of 400.** The out-of-fold U is overstated by roughly 3p/(training
  rows): about 13% at p = 14 with 320 training rows, against 4% at p = 4. That could hide a
  slow decline. A second block of 400 pool rows halves the bias.

**Checks, post hoc and labelled as such:** M4 re-evaluated with U_same on 800 rows (A2) and
with U_best on 800 rows (A1), same fit range (p ≥ 4) and the same bootstrap.

**Interpretation, fixed now:**
- **If M4 still fails under A1:** "mean-field doesn't catch up in the window up to p = 14"
  holds against the strongest mean-field student we have.
- **If M4 holds under A1 or A2:** the primary failure was due to the matched setting or the
  sample size, and we'll say that instead.

**Code:** `run.py banks --blocks 2 --menu`, then `run.py amend`. The output is
`S2_AMEND.csv`, `S2_AMEND_ALPHAS.csv`, `S2_AMEND_CHECKS.csv`. One test checks that with a
single block, A2 reproduces the primary U exactly.
