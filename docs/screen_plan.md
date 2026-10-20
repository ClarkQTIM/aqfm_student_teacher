# Screen plan: can we tell in advance which datasets need the exact map?

*Written 2026-10-20, before any of it was run. Committing this file before the runs is the
point: the checks and predictions in Section 6 are fixed here, and the git history shows they
came first. If we change any of them later, we add a dated note in Section 9 rather than editing
the original text.*

## 1. The question

The τ sweep (`docs/aqfm_overview.tex`) found a window, c ≥ 2, τ ≈ 3–20 with X/Y readout, where
labels made by the exact map can be learned by the exact map and by nothing classical we tried.
On a real dataset we don't know how the labels were made, and running the exact map on hardware
is expensive. So we want two stages:

1. **A screen:** cheap tests on (x, y) that say "this dataset might be worth it" or "don't bother".
2. **A pilot:** the exact map, run on the datasets the screen flags plus some it doesn't flag.

This plan covers building the screen and testing it on synthetic datasets where we know the
answer, before trying it on real data.

## 2. What "worth it" means

We use the same yardstick as the overview. A dataset is **worth a pilot** if

    margin = AUC(exact student) − max(every other student) ≥ 0.05,

where "every other student" means the classical curves, the 22 suite maps, the nonlinear learners
(HGB, RBF SVM, MLP), the mean-field and J=0 students, and the ground-state student.
In the current τ sweep, about 73 of the 864 datasets meet this: 55 of the 63 window datasets,
plus a few at c = 1 and at the edges of the window.

## 3. Two problems with the library as it stands

**Problem 1: every dataset has the same label noise.** All 864 teachers use temperature 0.25, so
the best AUC any model could get (we call this the *ceiling*; in the library it's the teacher's
own score, the "oracle AUC") is about 0.95 everywhere. That means "classical tops out at 0.79" can
only mean "there's signal classical can't reach." On real data a 0.79 usually means noisy labels.
A screen tuned on the current library will learn "low classical AUC means quantum helps," and
that rule fails on real data.

**Problem 2: hard for classical isn't the same as needs quantum.** Labels from the mean-field
teacher in the window are as hard for classical students as labels from the exact teacher, or
harder (classical/nonlinear 0.71 vs 0.79). But the mean-field student learns them almost
perfectly (0.95). So a test that only looks at (x, y) can at best say "there's signal nothing
classical reaches". It can't tell "the exact map will reach it" apart from "a mean-field model
will reach it".

## 4. Fixing the library: label-noise copies

For a subset of teachers we make two noisier copies of the labels, at temperature 0.5 and 1.0.
For a standardized score these give ceilings of about 0.86 and 0.74 (checked numerically; the
original 0.25 gives 0.95).

- **Subset:** exact and mean-field teachers with X/Y/Z (`local`) readout, c ∈ {0.5, 2, 8},
  τ ∈ {1, 3, 5, 10, 20, 40}, p ∈ {4, 6, 8}, giving 108 datasets. Add J=0 teachers with local
  readout, seed 0, the same τ values, giving 18. That's 126 per temperature, **252 new datasets**.
  The Z-only readout is left out, since it never produced a margin at any setting.
- **Paired:** each copy uses the original dataset's random seed, so the same weights w and the
  same random draws. Only the temperature changes. The T = 0.5 and T = 1.0 labels are then
  noisier versions of the T = 0.25 labels, so any change in the margin comes from noise alone.
  Names get a suffix, e.g. `tt_Q_p6_k2_t5_local_s0_T0.5`, but the seed comes from the name
  without it.
- **No new simulations.** The banks are already saved. We rerun the students and the nonlinear
  add-on on the 252 datasets, about 29% of the τ sweep's dataset count. Shot noise isn't needed
  here.
- **Where:** a new folder `runs/screen/` that links to `runs/tau_sweep/banks`. Nothing in
  `runs/tau_sweep` is changed.

After this the library has 864 + 252 = 1116 datasets, and in 378 of them the same teacher appears
at three noise levels.

## 5. The measures

Grouped by what they cost and what they're allowed to see.

### A. From the data alone (no map of any kind)

| | Measure | Already computed? |
|---|---|---|
| A1 | Best classical AUC: the max of the classical curves, suite maps and nonlinear learners | yes |
| A2 | Still rising? The AUC gain of the last doubling of the Fourier/RFF curve (64p → 128p) | yes (curve files) |
| A3 | Interaction ladder: AUC of an additive model (splines, no interactions) → HGB limited to pairs (`interaction_cst`) → unrestricted HGB. Reported as two gaps, pairs − additive and full − pairs | additive yes; the rest new |
| A4 | Ceiling estimates (new, see below) | no |
| A5 | Slot for further label-aware measures from the brainstorm | — |

**A4 is the weak point.** The rule in Section 6 hinges on an estimate of the ceiling, and I don't
have a method I trust at n = 600, p = 8. Candidates:
- **Nearest-neighbour bound.** The 1-NN error bounds the best possible error
  (Cover & Hart 1967). This bounds accuracy, not AUC, and it's biased at small n.
- **Repeat-fit spread.** How much the best learners' out-of-fold scores agree. This is
  speculative and needs thought.

The library helps here: we know the true ceiling of every dataset, so we can test each estimator
directly (check P4).

### B. Cheap classical stand-ins for the map

| | Measure | Already computed? |
|---|---|---|
| B1 | Best stand-in AUC: max(mean-field student, J=0 student, ground-state student) | yes |
| B2 | Best cheap AUC: max(A1, B1) | yes |

These are classical. Mean-field costs about as much as the classical models. The ground state
grows as 2^p, which is fine up to the sizes we can simulate.

### C. The exact map (the pilot, not the screen)

| | Measure |
|---|---|
| C1 | Centred kernel–target alignment (Cristianini et al. 2001; Cortes et al. 2012) of the exact-map kernel, minus the same for the best classical kernel |
| C2 | Huang et al. (2021): geometric difference between the classical and quantum kernels, plus the "model complexity" of the labels under each. This is the published test closest to what we're doing, so it's our baseline |
| C3 | Reduced-menu pilot: the margin using 9 families × 3 c × 4 τ = 108 map settings instead of 540 |
| C4 | Your earlier descriptors (class separation, kernel alignment, Gram spectrum, linear margin), on exact features. The same descriptors on mean-field features belong in B |

**Caveat for C:** on the synthetic library, C is partly circular. The labels were made by one of
the settings in the exact student's menu, so the exact features fit them by construction. We
report C, but a good C score here says little about real data.

### Rules about what each measure may see

- No measure in A or B may use exact-map features.
- No measure may use the teacher's score or the oracle AUC. Those are used only to *evaluate*
  measures, never as an input.
- Anything with a tuning step does its tuning inside cross-validation folds, the same way the
  students do.

## 6. Checks, written down before running

**The screen rule.** It's fixed now and has nothing fitted:

    flag the dataset  if  (ceiling estimate) − (best cheap AUC, B2) ≥ 0.05.

In words: there's at least 0.05 of AUC that nothing cheap reaches.

**How we score a screen.**
- Main score: AUROC for separating worth-it datasets from the rest.
- At a pilot budget of the top 10%: what fraction of the flagged datasets are worth it, and what
  fraction of the worth-it datasets get flagged.
- Everything is reported per p, and pooled. If we ever fit a combination of measures, it is fit on
  two p values and tested on the third, then repeated leaving each c out in turn.

**Caveat on the scores.** Datasets with the same p share their inputs, so they aren't independent.
Confidence intervals come from a bootstrap that resamples whole teacher settings, not single
datasets.

| Check | What | Pass if | Prediction | Blind? |
|---|---|---|---|---|
| **P1** | A1 alone (low classical AUC means flag), original noise only | — (reported) | Ranks the mean-field windows *above* the exact ones, because classical does worse on them (0.71 vs 0.79) | no: A1 is already in hand |
| **P2** | The screen rule with the **true** ceiling plugged in, all noise levels | AUROC ≥ 0.9 | Pass. This is the best any screen built from A + B can do; it tests the logic separately from the ceiling estimate | yes |
| **P3** | The screen rule with the **estimated** ceiling, all noise levels | AUROC ≥ 0.75 | Uncertain. This is the main question | yes |
| **P4** | Ceiling estimators vs the true ceiling | median abs. error ≤ 0.03 | Fails at p = 8 for nearest-neighbour estimates (they underestimate the ceiling) | yes |
| **P5** | Exact vs mean-field teachers inside the window: A alone, then A + B | A alone ≤ 0.5; A + B ≥ 0.9 | Both hold: data alone points the wrong way, and the stand-ins fix it | yes (new noise levels) |
| **P6** | C1, C2 as rankers | AUROC ≥ 0.85 | Pass, but partly circular (Section 5C) | yes |
| **P7** | Reduced-menu pilot (C3) vs the full margin call | same call on ≥ 85% | Pass | yes |
| **P8** | Worth-it count at each noise level | — (reported) | Drops sharply at T = 1.0: the window doesn't survive heavy label noise | yes |

**What we do with the outcome:**
- **P2 and P3 pass:** take the screen to real data (Section 7).
- **P2 passes, P3 fails:** the logic is right but we can't estimate the ceiling. On real data the
  screen then can't rule anything in. Every dataset goes to the pilot, so the pilot (C3) has to be
  cheap. We'd say this plainly.
- **P2 fails:** even with the true ceiling, cheap models plus a ceiling don't find the window.
  Back to the drawing board before any real data.

## 7. Real data (after the library)

- **Datasets:** the 55 real datasets, plus the heart replication. Before the archive is deleted,
  we need to confirm where these live now. They may be in `_archive_aqfm_2026-10-20`.
- **Mapping features to qubits** is an open decision. Options: pick the p features with the best
  single-feature AUC, selected inside the folds; or use the top p principal components. The same
  standardize-and-clip step as the synthetic inputs applies.
- **Screen:** run A + B on all datasets.
- **Pilot:** the top flagged datasets *plus* a few unflagged ones picked at random, e.g. 5 + 5.
  Without the unflagged ones we can't estimate how often the screen misses.
- **What a positive result looks like:** a real dataset flagged in advance, where the pilot then
  shows a margin ≥ 0.05 over every opponent. That's the outcome the project is after. Even then,
  it shows the data *behave like* window data. It doesn't show the data are quantum.

## 8. What gets built

- `experiments/ts_screen.py`, run in phases:
  - `noise-make`: the 252 label copies, into `runs/screen/`.
  - `noise-run`: the students and the nonlinear add-on on those copies. Reuses the τ-sweep code;
    restartable.
  - `measures`: A, B and C for all 1116 datasets, into `SCREEN_MEASURES.csv`.
  - `evaluate`: computes P1–P8 and prints pass/fail against the thresholds above.
- `tests/test_screen.py` checks that:
  - the noise copies use the same w and the same draws as the originals;
  - no A/B measure reads exact features, the teacher score or the oracle AUC;
  - the ceiling estimators behave on toy data with a known ceiling.
- `runs/tau_sweep` is never written to.

**Rough cost:** students on 252 datasets (about 29% of the τ sweep) plus the nonlinear add-on.
The measures take minutes to hours.

## 9. Open decisions and amendments

**Open now:**
1. Noise levels 0.5 and 1.0. Enough, or add 0.75?
2. A4: which ceiling estimators to try, and whether to brainstorm more A measures (A5) before
   building.
3. Real data: the feature-to-qubit rule, and the pilot size.

**Amendments:** none yet.

## References (to double-check before any write-up)

- T. Cover, P. Hart (1967), Nearest neighbor pattern classification, *IEEE Trans. Inf. Theory* 13(1):21–27.
- N. Cristianini, J. Shawe-Taylor, A. Elisseeff, J. Kandola (2001), On kernel-target alignment, *NeurIPS 14*.
- C. Cortes, M. Mohri, A. Rostamizadeh (2012), Algorithms for learning kernels based on centered alignment, *JMLR* 13:795–828.
- H.-Y. Huang et al. (2021), Power of data in quantum machine learning, *Nat. Commun.* 12:2631.
