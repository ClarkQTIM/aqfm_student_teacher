# Studies on the window

The window: exact teachers, full readout, coupling c ≥ 2, anneal time τ = 3–20. That's where
nothing cheap imitates the map (`docs/aqfm_overview.tex`, Table 3). Each study below asks one
question about it, and each has its own folder.

| | Study | Question | Status |
|---|---|---|---|
| S1 | [`s1_scaling/`](s1_scaling/PLAN.md) | As the number of qubits p grows, how much classical machinery does it take to match the map? And is that cost quantum, or does it come from the Ising-type map? | plan (draft) |
| S2 | [`s2_meanfield/`](s2_meanfield/PLAN.md) | Does mean-field catch up with the exact map as p grows, and how fast, in and out of the window? | primary run done (p = 2–14); Amendment 1 (A1, A2) ready |
| S3 | [`s3_coupling/`](s3_coupling/PLAN.md) | With other couplings (nearest-neighbour chain, random signs), does a window survive that mean-field and cheap classical methods can't close? | parked |
| S4 | [`s4_vqc/`](s4_vqc/PLAN.md) | Does a variational quantum circuit reach the map's performance with few parameters? That would show it has the right inductive bias where classical models don't | parked |

## How each study is kept tidy

- **One folder per study.** Each holds:
  - `PLAN.md`: the question, design and predictions. It's committed **before** anything is
    run, and changes go at its end as dated amendments.
  - `run.py`: the code, as phases (`make`, `run`, `score`, …).
  - `RESULTS.md`: written after the run. It quotes the commit the plan was fixed at.
- **Shared code lives in `studies/common/`.** That covers cell definitions, the teacher
  recipe, and the bank store. Studies **import** from `experiments/` and `aqfm_prescreen/` but
  never edit them, so every earlier result stays reproducible.
- **Outputs go to `runs/studies/<study>/`.** Simulations shared between studies go to
  `runs/studies/banks/`.
  - Banks are stored in blocks of 400 rows of one fixed input pool per p.
  - S2 uses block 0 and S1 extends the same files, so nothing is simulated twice.
- **Tests go in `tests/test_studies_<name>.py`.**
- **Same rules as before:**
  - at most 31 workers, under `taskset -c 0-31`;
  - nothing written to the home folder;
  - every run block comes with copy-paste commands and STOP checks.
