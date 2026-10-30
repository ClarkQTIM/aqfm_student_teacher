# Where we are

*Updated 2026-10-29. One page; details live in the linked write-ups.*

**The question for the paper:** is there a setting where the annealed quantum feature map
(AQFM) gives labels that classical methods can't learn, and is that because of the quantum
part?

## The two questions (how every result below fits together)

| Question | Answer in the window | Quantum? | Results |
|---|---|---|---|
| **A. Can a generic classical learner learn these labels from examples?** | **No**, for exact-map labels *and* mean-field labels; mean-field's are even harder | **No.** Mean-field is classical and shares the difficulty, so it comes from the coupled anneal dynamics. | 4, 5 (also geometry G5) |
| **B. Can mean-field, the cheap classical twin, reproduce what the exact map does?** | **No.** It misses about 80% of the exact map's labels, at every size up to 16 qubits | **Yes.** The only thing mean-field leaves out is the quantum correlations. | 1, 2, 3 |

**In one sentence:** both kinds of label are hard for classical learners, but only the exact map
does something its classical twin can't copy. The quantum part makes the labels *different* from
mean-field's, not *harder to learn*.

**Expected vs surprising.**
- **Expected, and found:** the window (1), the geometry (2), fixed bases failing, and networks or
  trees doing better but still falling short (5).
- **Surprises:**
  - mean-field never catches up (3), where we predicted slow convergence;
  - polynomial difficulty doesn't grow with qubits (4);
  - mean-field labels are harder for learners than exact ones (5), where we predicted they'd
    be about equal.

## Done

| | Finding | Where |
|---|---|---|
| 1 | **A window exists** (coupling c ≥ 2, anneal time τ = 3–20, full readout). In it, the exact map's labels beat every opponent, including its own mean-field twin. | teacher–student and τ sweep: `docs/aqfm_overview.tex` |
| 2 | **The geometry agrees.** A label-free measure from theory (Huang et al. 2021) picks out the same window: AUROC 0.95. | `docs/geometry_results.md` |
| 3 | **Mean-field doesn't catch up** as qubits are added. Inside the window it still misses about 80% up to 16 qubits; outside it, it becomes exact. | `studies/s2_meanfield/RESULTS.md` |
| 4 | **Polynomials stay flat in the window,** even with 30,000 rows and no noise. Their difficulty is set by the anneal time, not by the qubit count. | `studies/s1_scaling/RESULTS.md` (Part A) |
| 5 | **Networks and boosted trees fall short too** in the window, somewhat more as qubits are added (0.02 → 0.06–0.11 AUC). **But mean-field labels are even harder for them**, so that difficulty comes from the anneal, not the quantum part. The quantum claim rests on result 3. | `studies/s1_scaling/RESULTS.md` (Part B) |

## Left for the paper

| | What | Why | Status |
|---|---|---|---|
| 1 | **Robustness check for the networks** (longer training, two learning rates; p = 8) | Rules out "the networks were undertrained" (question A) | **done**: no change (0.077 → 0.077); R1, R2 held |
| 2 | **One stronger classical twin** than mean-field (pair or cluster corrections) | The quantum claim rests on question B, so a reviewer will ask whether mean-field was just too weak a twin | first run done (D2 held at 0.68, but DTWA not converged in M; at c = 2 DTWA catches up); **convergence run on GPU (Amendment 2), 2026-10-30** |

## Next paper

| | What | Status |
|---|---|---|
| 3 | **Variational quantum circuit** (S4): does the right quantum structure help, with few parameters? Generic circuits vs ones built from the map's own Hamiltonian (circuit diagram: `studies/s4_vqc/figures/fig_vqc_circuits.png`) | plan final (predictions V0–V4); code ready, to run |

## Parked (doesn't change the paper's claim)

- S3: other couplings (`studies/s3_coupling/PLAN.md`).
- Tensor networks.
- Part A tidy-ups (interaction-order profile; QR solver for degree ≥ 8 at p ≤ 5).
- Geometry G7/G8 redesigns.
- The real-data search for labels that use the window.

## Key references

1. Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H. & McClean, J. R. (2021). Power of data in quantum machine learning. *Nature Communications* 12, 2631. [arXiv:2011.01938](https://arxiv.org/abs/2011.01938)
2. Schachenmayer, J., Pikovski, A. & Rey, A. M. (2015). Many-body quantum spin dynamics with Monte Carlo trajectories on a discrete phase space. *Physical Review X* 5, 011022. [arXiv:1408.4441](https://arxiv.org/abs/1408.4441)
3. Schuld, M., Sweke, R. & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A* 103, 032430. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
4. Landman, J., Thabet, S., Dalyac, C., Mhiri, H. & Kashefi, E. (2023). Classically approximating variational quantum machine learning with random Fourier features. *ICLR 2023*. [arXiv:2210.13200](https://arxiv.org/abs/2210.13200)
5. Sweke, R., Recio-Armengol, E., Jerbi, S., Gil-Fuster, E., Fuller, B., Eisert, J. & Meyer, J. J. (2025). Potential and limitations of random Fourier features for dequantizing quantum machine learning. *Quantum* 9, 1640. [arXiv:2309.11647](https://arxiv.org/abs/2309.11647)

*Each write-up lists its own references in full. Two physics statements in S2 still need sources (see `studies/s2_meanfield/RESULTS.md`).*

## Limits to keep in view

- **Home court.** The labels come from the map, so the claim is "generic learners can't learn
  labels of this kind". On real data, nothing has won yet.
- **Hard for classical isn't the same as quantum.** Mean-field labels are also hard for
  classical models in the window. The quantum claim rests on the exact-versus-mean-field gap.
- **Only simulated systems** (up to 16 qubits), which a classical computer can simulate.
