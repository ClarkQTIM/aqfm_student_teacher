# S5: Does a stronger classical twin copy the exact map?

*Plan, FINAL 2026-10-29, before any S5 code has been written or run. Predictions D0–D4 are in
Section 5; D2 is the user's bet, option (a).*

## 1. Why

The quantum claim now rests on one result (question B in `STATUS.md`): in the window,
mean-field can't reproduce the exact map. It misses about 80% of the exact labels at every
size up to 16 qubits (S2).

Mean-field is the *weakest* classical twin, because it sets every correlation between qubits
to zero. S2 measured what that throws away: pair correlations that are small one by one but
share a sign, so their total pull on each qubit stays large (C_coll ≈ 2.3).

A reviewer will ask: **would a classical method that keeps those correlations copy the exact
map?**
- **If no,** the quantum claim becomes much stronger. The gap survives a serious classical
  attempt to put the correlations back.
- **If yes,** the window's "quantum part" is cheap to simulate classically (dequantized). We
  need to know that before the paper, not after.

## 2. The stronger twin: discrete truncated Wigner approximation (DTWA)

**Why this method (literature).**
- **The method:** Schachenmayer, Pikovski & Rey, *Phys. Rev. X* 5, 011022 (2015),
  [arXiv:1408.4441](https://arxiv.org/abs/1408.4441). They introduced DTWA and benchmarked it
  against exact results on Ising models with all-to-all interactions and with a transverse
  field. It reproduces single-spin values, pair correlations and squeezing.
- **The framework:** Polkovnikov, "Phase space representation of quantum dynamics", *Ann. Phys.*
  325 (2010), [doi:10.1016/j.aop.2010.02.006](https://doi.org/10.1016/j.aop.2010.02.006).
  The standard review of truncated-Wigner methods.
- **Accuracy is best for all-to-all coupling:**
  - Khasseh, Russomanno, Schmitt, Heyl & Fazio, *Phys. Rev. B* 102, 014303 (2020),
    [arXiv:2004.09812](https://arxiv.org/abs/2004.09812). For transverse-field Ising models,
    DTWA agrees well with exact results at all-to-all range, and less well as range shortens.
  - Kunimi, Nagao, Goto & Danshita (2021), [arXiv:2008.13481](https://arxiv.org/abs/2008.13481).
    DTWA's validity time grows with interaction range.
- **It has been used against quantum-annealing claims:** Sels (2026),
  [arXiv:2606.20187](https://arxiv.org/abs/2606.20187), a preprint not yet peer-reviewed. It
  simulates near-adiabatic annealing of spin glasses classically with DTWA, as a challenge to a
  beyond-classical claim, and reports accuracy improving with system size.

**So DTWA is the opponent a physicist reviewer would choose for our setup, and all-to-all
annealing is its best case.** That makes it a hard test, and the right one.

**How it works:**
- **Mean-field** follows one classical spin per qubit, a unit vector (the Bloch vector). Each
  spin precesses in the field from its input and the *average* of the other spins.
- **DTWA** follows **many** such classical systems ("trajectories"). Each starts from a
  random point that mimics quantum uncertainty: for our start state |+⟩, s_x = +1, and s_y,
  s_z = ±1 at random (vectors of length √3).
  - Each trajectory evolves under the same classical equations as mean-field, with **its own**
    field: h_i = x_i + J Σ_{j≠i} s_z^j for that trajectory.
  - Averaging over trajectories gives ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ and the pair correlations.
- **Why it can do better:** trajectories that drift apart carry the correlations mean-field
  drops.
- **Cost:** mean-field × the number of trajectories M (hundreds to thousands). That's still
  polynomial in p, so classical and cheap.

**Built-in checks of the code:**
- With one trajectory started from the unit vector (1, 0, 0), DTWA **is** mean-field. The test
  compares it with `meanfield_local` to 10⁻⁹.
- With J = 0, DTWA averaged over all starting points is **exact** (the equations are linear).
  The test compares it with `single_qubit_strang` to 10⁻⁹.

**Why DTWA and not the pair (cumulant) expansion.** DTWA reuses mean-field's integrator and
is the method physicists usually reach for here. The pair expansion (track every pair
correlation, assume three-way correlations vanish) is the fallback if DTWA turns out
ambiguous.

## 3. Design

Same as S2, so the numbers compare directly:
- **Qubits:** p ∈ {2, 3, 4, 5, 6, 8, 10, 12}, and 14 if the cost allows.
- **Cells:** S2's 14. That's 12 window cells (c ∈ {2, 4, 8} × τ ∈ {3, 5, 10, 20}) and the two
  controls (c = 0.5, τ = 5) and (c = 4, τ = 1).
- **Inputs:** block 0 of the shared pool (400 rows per p). The exact banks already exist from
  S2, so **no new quantum simulation is needed.**
- **Trajectories:**
  - **Fixed seed per (p, cell, row).** The same random starts in every run.
  - **Exact enumeration** of all 4^p starting points when 4^p ≤ M. That's p ≤ 5 at M = 1,024.
  - **Sampling-noise check:** every cell is also run at M/4. A trajectory average carries noise
    of about 1/√M per number, and noisy features make any copy look worse.
    - If U at M and at M/4 differ by more than 0.02 in a cell, that cell is rerun at 4M.
    - Every value is reported with both M.
- **M = 1,024 to start.** Measured cost (Section 6) may raise it.
- **Matched setting only:** the DTWA twin uses the teacher's own (c, τ). S2's amendment showed
  that letting mean-field pick its best of 60 settings changed U by only about 0.1. It can be
  added later if the result is close.

## 4. What we measure (exactly S2's measures, with the DTWA twin in place of mean-field)

- **U_DTWA, the headline.** How much of an exact teacher's noise-free score the DTWA features
  (⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit, 3p numbers) can't explain.
  - S2's recipe: 20 random readouts; 5-fold cross-validated least squares; median of 1 − R².
  - **Same rows, same readouts, same folds as S2's U**, so U_DTWA and U_MF compare row for row.
- **E_DTWA:** RMS distance between exact and DTWA Bloch vectors.
- **Correlations:** DTWA's pair correlations against the exact ones. That's the correlation
  between per-pair values, and C_coll from each side. This shows directly whether DTWA puts
  back what mean-field drops.
- **Exponent α** for U_DTWA and E_DTWA against p (fitted on p ≥ 4, bootstrap over rows), as in
  S2.

## 5. Predictions (fixed when this draft is committed)

| | Prediction |
|---|---|
| **D0** | **Sanity, must hold:** in the short-anneal control (c = 4, τ = 1), U_DTWA ≤ U_MF at every p, and U_DTWA < 0.01 at p ≥ 8 |
| **D1** | **DTWA beats mean-field in the window:** U_DTWA < U_MF in at least 9 of 12 window cells at every p |
| **D2** | **The quantum part survives DTWA** (the user's bet): the median U_DTWA over the 12 window cells at p = 12 is **above 0.3** (at the larger M if the noise check triggered). Below 0.1 would mean DTWA largely copies the exact map: the window's quantum part is classically cheap |
| **D3** | **Trend:** the median α of U_DTWA over window cells, with its interval, *reported*. A positive α (DTWA catching up as p grows) matters most, because DTWA is a semiclassical method that becomes more accurate as the number of spins grows |
| **D4** | *Reported:* E_DTWA, the correlation match, and U_DTWA by anneal time |

**My expectation, for the record.** DTWA beats mean-field clearly (D1), most at the short end
of the window (τ = 3–5), but leaves a large share unexplained at τ = 10–20. That's S2's
picture: at long anneals, fluctuations grow around an unstable trajectory, and that's where
semiclassical methods tend to struggle. It's reasoning, not a measurement, and it's the
reason D2 is a real bet.

## 6. Cost (to be measured by a `time` phase before the run)

- **Per (p, cell):** 400 rows × M trajectories × p spins × n_steps, where n_steps = 160τ, so up
  to 3,200 at τ = 20.
  - At p = 12, M = 1,024 and τ = 20, that's about 1.6 × 10¹⁰ spin updates.
  - That's minutes on a GPU, and slow on CPU.
- **Plan:** the integrator in PyTorch on GPU (like `torchsim.py`), with a NumPy version for
  the tests. A `time` phase measures one cell first.
- **Any machine with a recent driver works.** ophlslcec08 (2 × H100) works; ophlappc05 does
  not (CUDA 12.6 driver).

## 7. What this doesn't do

- **Other classical twins** (the pair expansion, tensor networks) aren't tested; the pair
  expansion is the fallback.
- **No learners.** That's S1.
- **No other couplings.** That's S3.
- **The range is p ≤ 12–14.** Exponents describe that range only.

## 8. Changes made while writing the code (before any run on real data)

- **CPU, not GPU (Section 6).** The NumPy integrator turned out fast enough: about 16 minutes
  for one 12-qubit cell at τ = 20 with M = 1,024 on one core. That's roughly 15–30 minutes for
  everything on 31 cores, so there's no PyTorch version.
- **Qubits:** S2's full list, p ∈ {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}. S2's exact banks exist
  for all of them. p = 14 and 16 are included if `time` shows they're affordable. D2 is still
  tested at p = 12.
- **The M/4 check uses the first quarter of the same trajectories** (nested), so the two values
  differ only by sampling. Cells with every starting point enumerated (4^p ≤ M) have no sampling
  noise and are never flagged.
- **No prediction changed.**

**Disclosure.** While checking the code, I compared DTWA with the exact map on 20 random inputs
(not the study's pool) at p = 4, c = 4, τ = 10, a window-like setting. DTWA's RMS error on the
single-qubit readouts was 0.10, against mean-field's 0.42. That look came after the predictions
were committed and pushed (e65eada), and they are unchanged.

**Code:** `studies/common/dtwa.py` (the method), `studies/s5_twin/run.py` (phases `time`, `banks`,
`measure`, `score`). Tests: `tests/test_studies_s5.py`. Outputs: `runs/studies/s5/`.

## 9. Amendment 1 (2026-10-29, AFTER seeing the first scoring at M = 1,024)

**What the first scoring showed** (`runs/s5.log`; provisional):
- **D2:** median window U_DTWA at p = 12 was 0.663 (mean-field 0.818).
- **D0 and D1 failed, but only at small p.** On the short-anneal control, DTWA was worse than
  mean-field at p = 3–6 (with all starting points enumerated at p ≤ 5, so this isn't sampling
  noise). It was essentially exact from p = 10.
- **The noise check flagged many cells,** most at c = 2. There, U at M/4 exceeded U at M by up
  to 0.19.

**1. Coupled-case code check (added).** The two original checks don't involve the coupling. A
new test runs fields + coupling with no driver (s held at 1). In that case DTWA's single-qubit
values are exact, because every s_z is conserved and averaging over s_z = ±1 gives the exact
product of cosines. `dtwa.evolve` matches the exact values to 10⁻¹⁰ at p = 2, 3, 4 and 6. So
the coupling code (sign, factor 2, field from the other spins) is right. The small-p failures of
D0 and D1 are DTWA's own limitation, not a bug. That's consistent with its accuracy improving
as each spin interacts with more others [3, 4].

**2. Noise: the plan's rule is applied as written.** Every flagged cell is rerun at 4M = 4,096
(`banks --flagged --M 4096`), and the checks are rescored on the larger-M banks.
- **The bias works for the quantum claim:** noisy DTWA features lower R², so U looks higher.
  That's why it has to be removed before D2 is read.
- **Reported, not a test:** U extrapolated to infinitely many trajectories, (4·U(M) − U(M/4))/3.
  That's the standard first-order correction if the bias shrinks like 1/M. If the extrapolated
  value at p = 12 is close to 0.3, D2 is reported as unsettled, whatever the test gives.

**No prediction changed.**

## References

1. Schachenmayer, J., Pikovski, A. & Rey, A. M. (2015). Many-body quantum spin dynamics with Monte Carlo trajectories on a discrete phase space. *Physical Review X* 5, 011022. [arXiv:1408.4441](https://arxiv.org/abs/1408.4441)
2. Polkovnikov, A. (2010). Phase space representation of quantum dynamics. *Annals of Physics* 325, 1790. [doi:10.1016/j.aop.2010.02.006](https://doi.org/10.1016/j.aop.2010.02.006)
3. Khasseh, R., Russomanno, A., Schmitt, M., Heyl, M. & Fazio, R. (2020). Discrete truncated Wigner approach to dynamical phase transitions in Ising models after a quantum quench. *Physical Review B* 102, 014303. [arXiv:2004.09812](https://arxiv.org/abs/2004.09812)
4. Kunimi, M., Nagao, K., Goto, S. & Danshita, I. (2021). Performance evaluation of the discrete truncated Wigner approximation for quench dynamics of quantum spin systems with long-range interactions. [arXiv:2008.13481](https://arxiv.org/abs/2008.13481)
5. Sels, D. (2026). Truncated Wigner dynamics of biclique quantum spin glasses. Preprint, not yet peer-reviewed. [arXiv:2606.20187](https://arxiv.org/abs/2606.20187)
