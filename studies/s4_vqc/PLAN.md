# S4: Does a variational quantum circuit have the right inductive bias?

*Plan, FINAL 2026-10-29, before any S4 code has run on real data. It replaces the parked sketch of
2026-10-26. Design decisions (user, 2026-10-29): learn from the labels; L = 1, 2, 4, 8, 16;
both shared and per-qubit informed circuits; p = 4, 6, 8.*

## 1. The question

In the window, generic classical learners can't learn the map's labels (S1), and the exact map's
cheap classical twin can't reproduce them (S2). **Does a trainable quantum circuit with few
parameters reach the exact map's performance?** And if it does, is that because it's quantum,
or because it has the right physics built in?

We look for **parity**: test AUC within 0.01 of the open-book student (logistic regression on
the teacher's own map features), with far fewer parameters than the classical learners use.

## 2. The circuits

See `figures/fig_vqc_circuits.png` for both drawn at 3 qubits. Every circuit is read out the
same way as the map, as ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on each qubit (3p numbers), followed by a trainable linear
layer and a logistic output.

| | Student | Each layer | Circuit parameters |
|---|---|---|---|
| **G1** | **generic, inputs encoded once** [1] | RY(θ) and RZ(θ) on every qubit (own angles), then a CNOT ring; inputs enter once as RY(x_i) | 2pL |
| **G2** | **generic, inputs re-encoded every layer** [2] | RY(x_i) on every qubit, then as G1 | 2pL |
| **H1** | **Hamiltonian-informed, shared** (QAOA-style) [3] | e^{i(γ_ℓ Σ x_i Z_i + η_ℓ Σ_{i<j} Z_i Z_j)} then e^{iβ_ℓ Σ X_i}, starting from \|+⟩: the map's own three pieces with trainable times | 3L |
| **H2** | **Hamiltonian-informed, per qubit** | as H1, with a separate data strength γ_{ℓ,i} and driver time β_{ℓ,i} on each qubit, and one coupling η_ℓ | (2p + 1)L |

- **The informed circuits know the map's ingredients, not its setting.** They're given neither
  the coupling c nor the anneal time τ: η and γ are learned separately.
- **The real anneal is circuit H1 with about 1,600 tiny, fixed layers.** A test checks this
  identity exactly (Section 6).
- **L ∈ {1, 2, 4, 8, 16} layers** for every student. Parameters run from 3 (H1, L = 1) to 256
  (G1/G2 at p = 8, L = 16), plus the readout's 3p + 1.

## 3. Data, training and the comparison

- **Teachers:** S1's, all 99 per p (exact and mean-field at the 14 cells, J = 0 at 5 anneal
  times; 3 readout seeds each), at **p = 4, 6, 8**.
- **Rows:** exactly S1 Part B's splits. The first n rows of the pool; 80% train, 20%
  validation; test = rows 30,000–39,999.
  - **n = 3,000** for every p (the primary test);
  - **n = 30,000** at p = 8 (matches Part B's main number).
- **Training,** the same for every student:
  - Adam, learning rate 0.01, batch 256;
  - validation loss every 25 steps, early stopping with patience 40 checks, at most 10,000
    steps;
  - each student keeps its best validation checkpoint;
  - every teacher gets its own circuit, all trained in parallel on GPU.
- **Starting values:**
  - **generic:** angles uniform on [0, 2π);
  - **informed:** the standard linear-ramp start for QAOA-style circuits [4], a discretized
    anneal of total time T₀ = 5 with coupling c = 1. That means γ_ℓ = s_ℓ T₀/L,
    β_ℓ = (1 − s_ℓ) T₀/L, η_ℓ = γ_ℓ/(p − 1), with s_ℓ = (ℓ − ½)/L. Training moves them from
    there.
- **What it's compared with** (from `S1_PARTB.csv`, same teachers and rows):
  - the open-book student;
  - Part B's best classical learner (networks or trees, chosen on validation).
- **Choosing L:** within each student, L is chosen on validation loss, as Part B chose sizes.
  "Reaches parity at L" is read off afterwards.

## 4. Predictions (fixed now; tested at p = 8, n = 3,000; window = exact teachers in the 12 window cells; cell value = median over the 3 readout seeds)

| | Prediction |
|---|---|
| **V0** | **Sanity, must hold:** H1 at L = 16 gets within 0.01 of the open-book student in the short-anneal control (c = 4, τ = 1), exact teachers, at p = 4, 6 and 8. That anneal is short, so 16 layers approximate it well. A failure means the training is broken |
| **V1** | **Generic circuits are no better than classical learners:** the better of G1 and G2 has AUC ≤ Part B's best classical learner + 0.01 in at least 9 of 12 window cells. Encoding once limits the frequencies a circuit can make [2, 5], and re-encoded circuits are often matched by random Fourier features [6, 7] |
| **V2** | **The informed circuit reaches parity:** the better of H1 and H2 gets within 0.01 of the open-book student in at least 6 of 12 window cells, with at most 16 layers |
| **V3** | **Structure beats generic:** H1 has a higher AUC than the better generic circuit in at least 9 of 12 window cells, despite fewer parameters at every L |
| **V4** | *Reported, no prediction:* the smallest L that reaches parity, by cell and p; H1 against H2 (does more freedom help?); n = 30,000 at p = 8; and the same for **mean-field teachers** and J = 0 teachers |

**Why the mean-field teachers matter (V4).** The informed circuit is partly on home ground: it
shares the teacher's ingredients. Mean-field labels come from the same ingredients without the
quantum correlations. So if H1 learns exact labels well but mean-field labels poorly, its
advantage is tied to the quantum dynamics it imitates, not just to the right kind of wiggles.

**What the outcomes mean:**
- **V2 and V3 hold:** the right physics structure is the inductive bias. A few dozen parameters
  of the right shape do what thousands of generic parameters can't.
- **V1 fails (generic circuits reach parity too):** being a quantum circuit is enough. That's
  surprising, and it's the outcome that would change the story most.
- **V2 fails:** compressing the anneal into ≤ 16 trainable layers isn't enough. The map's long
  schedule matters.

## 5. Cost (measured by a `time` phase before the run)

- **Jobs:** 4 students × 5 depths × 3 p at n = 3,000, plus 20 at p = 8, n = 30,000. That's
  80 jobs, each training 99 circuits at once.
- **Size:** statevectors of 2^p ≤ 256 amplitudes, batched over teachers and rows on GPU. A
  `time` phase measures the slowest job (G1 or G2 at p = 8, L = 16, n = 30,000) first.
- **Machine:** any with a recent GPU driver (ophlslcec08 works).

## 6. Code checks before any run

- **The informed circuit reproduces the anneal.** With its layers set to the map's own Strang
  steps, H1 must give exactly the exact map's bank (compute_bank, `strang`) to 10⁻⁶.
- **The PyTorch circuits match a plain NumPy version** of the same gates, to 10⁻⁵. The NumPy
  version is the one tested against the anneal locally.
- **Gradients flow:** a tiny training run lowers the training loss.

## 7. What this doesn't do

- **No shot noise:** exact expectation values.
- **No hardware noise.**
- **p ≤ 8:** VQC training cost grows like 2^p.
- **Not a test of quantum computers:** everything is classically simulated. It asks which
  structure learns these labels, not what a device would do.

## References

1. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M. & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *Nature* 549, 242–246. [arXiv:1704.05018](https://arxiv.org/abs/1704.05018)
2. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E. & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. *Quantum* 4, 226. [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)
3. Farhi, E., Goldstone, J. & Gutmann, S. (2014). A quantum approximate optimization algorithm. [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
4. Zhou, L., Wang, S.-T., Choi, S., Pichler, H. & Lukin, M. D. (2020). Quantum approximate optimization algorithm: performance, mechanism, and implementation on near-term devices. *Physical Review X* 10, 021067. [arXiv:1812.01041](https://arxiv.org/abs/1812.01041)
5. Schuld, M., Sweke, R. & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A* 103, 032430. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
6. Landman, J., Thabet, S., Dalyac, C., Mhiri, H. & Kashefi, E. (2023). Classically approximating variational quantum machine learning with random Fourier features. *ICLR 2023*. [arXiv:2210.13200](https://arxiv.org/abs/2210.13200)
7. Sweke, R., Recio-Armengol, E., Jerbi, S., Gil-Fuster, E., Fuller, B., Eisert, J. & Meyer, J. J. (2025). Potential and limitations of random Fourier features for dequantizing quantum machine learning. *Quantum* 9, 1640. [arXiv:2309.11647](https://arxiv.org/abs/2309.11647)
8. Barends, R. et al. (2016). Digitized adiabatic quantum computing with a superconducting circuit. *Nature* 534, 222–226. [arXiv:1511.03316](https://arxiv.org/abs/1511.03316) (the anneal run as a gate circuit, as in Section 2)
