# Where we are

*Updated 2026-10-28. One page; details live in the linked write-ups.*

**The question for the paper:** is there a setting where the annealed quantum feature map
(AQFM) gives labels that classical methods can't learn, and is that because of the quantum
part?

## Done

| | Finding | Where |
|---|---|---|
| 1 | **A window exists** (coupling c ≥ 2, anneal time τ = 3–20, full readout). In it, the exact map's labels beat every opponent, including its own mean-field twin. | teacher–student and τ sweep: `docs/aqfm_overview.tex` |
| 2 | **The geometry agrees.** A label-free measure from theory (Huang et al. 2021) picks out the same window: AUROC 0.95. | `docs/geometry_results.md` |
| 3 | **Mean-field doesn't catch up** as qubits are added. Inside the window it still misses about 80% up to 16 qubits; outside it, it becomes exact. | `studies/s2_meanfield/RESULTS.md` |
| 4 | **Polynomials stay flat in the window,** even with 30,000 rows and no noise. Their difficulty is set by the anneal time, not by the qubit count. | `studies/s1_scaling/RESULTS.md` (Part A) |

## Left for the paper

| | What | Why | Status |
|---|---|---|---|
| 1 | **Neural networks and gradient-boosted trees** (S1 Part B), from 300 to 30,000 rows | The biggest remaining threat. If they get there with enough data, the gap is a small-data effect. | predictions fixed (`studies/s1_scaling/PLAN.md` §9); code ready; **run 2026-10-28** |
| 2 | **One stronger classical approximation** than mean-field (pair or cluster corrections) | A reviewer will ask whether mean-field was too weak a twin | not started |

## Next paper

| | What | Status |
|---|---|---|
| 3 | **Variational quantum circuit** (S4): does the right quantum structure help, with few parameters? | plan sketched (`studies/s4_vqc/PLAN.md`); needs architecture decisions |

## Parked (doesn't change the paper's claim)

- S3: other couplings (`studies/s3_coupling/PLAN.md`).
- Tensor networks.
- Part A tidy-ups (interaction-order profile; QR solver for degree ≥ 8 at p ≤ 5).
- Geometry G7/G8 redesigns.
- The real-data search for labels that use the window.

## Limits to keep in view

- **Home court.** The labels come from the map, so the claim is "generic learners can't learn
  labels of this kind". On real data, nothing has won yet.
- **Hard for classical isn't the same as quantum.** Mean-field labels are also hard for
  classical models in the window. The quantum claim rests on the exact-versus-mean-field gap.
- **Only simulated systems** (up to 16 qubits), which a classical computer can simulate.
