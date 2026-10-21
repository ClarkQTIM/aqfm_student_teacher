# Real-data plan: the ladder and the feature count on tabular ophthalmic datasets

*Written 2026-10-21, before any real dataset has been modelled. As with `screen_plan.md`,
committing this first is the point: the rules, splits and predictions below are fixed, and
changes go in Section 10 as dated amendments. Dataset sources and pitfalls are listed in
`claude/ophtho_datasets.md` (project).*

## 1. What we want to know

For each real dataset, two questions:

1. **The ladder.** Does the exact map beat everything cheap? Here "cheap" means classical
   features, nonlinear learners, mean-field and J=0. If it does, does that survive on data held
   back from the start?
2. **The feature count.** Even if classical models get there, how many features does it take
   them, compared with how many the map uses? At the *same* feature count, who is ahead?

The second question gives a result even if the map never wins. It's also much less exposed to
luck than chasing a margin.

## 2. What "fair" means for the feature count

- **The map's feature count is D_Q**: the number of measured expectation values in the readout
  the exact student chose, *not* the number of qubits. At p = 8 that is 8 (one Pauli type), 24
  (X, Y, Z per qubit), 28 (one type of pair), 84 (all pairs) or 108 (everything).
- **D\***: the smallest classical expansion (Fourier, random Fourier, polynomial, spline) whose
  AUC gets within 0.01 of the exact map's, or ∞ if none does. The ratio **R = D\* / D_Q** is
  "how many times more features classical needs".
  - D* is picked using the folds' test AUCs, which favours classical slightly. Any gap in the
    map's favour is therefore a conservative estimate.
- **Equal-count gap G**: AUC(exact) − the best classical expansion with **D ≤ D_Q**. Polynomial
  degree 1 is the raw inputs themselves (D = p), so plain linear regression on the same inputs is
  always included.
- **The same comparison against mean-field.** Mean-field features have exactly the same count
  as the exact ones and are cheap to compute classically. If mean-field matches the exact map at
  equal count, the compactness belongs to the *Ising-type map*, not to anything quantum. We
  report both:
  - **map compactness:** the better of exact and mean-field, vs classical at D ≤ D_Q;
  - **quantum part:** exact vs mean-field.
- **What "fewer features" does and doesn't buy.** Classical features cost almost nothing to
  compute. Each quantum feature is an expectation value needing about 1000 measurements (shot
  study). So this is a claim about **compactness** (label-relevant structure in fewer numbers,
  hence a smaller model), never about compute. We'll say so in any write-up.

## 3. Datasets and labels (fixed here)

The dataset list and the labels freeze when `run` is first executed on any dataset. Until then,
adapters can be finished after looking at the files' *columns*, but nothing is modelled.

| Dataset | Label (1 = positive) | Rows used | Groups for splitting | Dropped columns |
|---|---|---|---|---|
| DR Debrecen (UCI) | DR signs | all | none (one image per row) | none |
| PAPILA | glaucoma vs healthy; **suspects dropped** | glaucoma + healthy eyes | **patient** | mean defect (recorded only for glaucoma → leaks the label), IDs |
| Myopia (Orinda) | myopic within 5 years | all | none | `id`, `studyyear` (administrative) |
| GRAPE | progression by PLR2 (most positives: 40) | baseline sheet | **patient** | the other progression labels (PLR3, MD slope), all follow-up visits, image references |
| Keratoconus Pentacam (Mendeley) | keratoconus vs normal (*finalize once downloaded*) | *finalize* | patient if given | any index the label is defined from |
| PERG-IOBA | any diagnosis vs normal | first visit per subject | subject | free-text comments; features = P50/N95 amplitudes and latencies per eye |
| FIVES / VascX (ours) | disease vs normal (*finalize from the file*) | all | patient if given | IDs |

Datasets that turn out to be unusable (too few positives in either half, licence) are dropped,
with the reason recorded, before `run`.

## 4. From the raw table to the qubits

Applied identically to every dataset, in this order:

1. **Clean.** Drop the columns above. Drop any feature with > 20% missing values. Encode binary
   categoricals as 0/1, and one-hot encode categoricals with more than two levels.
2. **Split once, 50 / 50:** a **discovery half** and a **confirmation half**.
   - The split is stratified by label and grouped by patient where there are groups.
   - The seed comes from the dataset name.
   - The confirmation half isn't looked at until the confirmation phase.
3. **Impute** the remaining missing values with the discovery half's medians.
4. **Choose the qubits.**
   - If the dataset has at most 8 usable features, use all of them: p = their number, as long
     as p ≥ 4; otherwise the dataset is dropped.
   - If it has more, use the **8 with the highest random-forest importance** (500 trees, seed
     0), fitted on the **discovery half only**.
   - Random-forest importance was chosen over single-feature AUC because it picks up features
     that matter only in combination, which is what the map might use.
   - The selected features and both rankings are saved.
5. **Scale.** Standardize with the discovery half's means and standard deviations (applied to
   both halves), and clip to ±3. This is the same input pipeline as the synthetic work.

Because the qubit set is chosen on the whole discovery half, the cross-validation scores *inside*
the discovery half are slightly optimistic for every student alike. The confirmation half is
untouched by any choice, which is why the claims are made there.

## 5. Students

These are exactly the τ-sweep students, on the p selected inputs:
- **Exact map, S_Q:** 9 readout families × c ∈ {0.5, 1, 2, 4, 8} × 12 anneal times. Inner
  cross-validation picks the setting and the regularisation.
- **Mean-field, S_MF**, the same menu; **J=0**, 9 × 12; **ground state**, 15.
- **classical_best:** the classical expansion curves plus the 22 suite maps, chosen by inner
  cross-validation.
- **Nonlinear learners:** boosted trees, RBF SVM and MLP on the p inputs, with the same grids as
  the τ-sweep add-on.
- **Reference only:** the same nonlinear learners on **all** cleaned features. This answers the
  practical question of whether the map on 8 chosen features beats just using everything. It is
  reported, but isn't part of any verdict.

**Simulation.**
- **Banks:** computed once per dataset on all rows. A bank is a row-by-row function of the
  inputs, with nothing fitted.
- **Label-free transforms:** the classical expansions (spline knots, random features) are fitted
  on all rows' inputs. They never see labels.

## 6. Discovery half: the ladder verdict and the feature count

Nested cross-validation, 5 outer × 3 inner folds, exactly as in the τ sweep. Where there are
groups (two eyes per patient), **both the outer and inner folds are grouped by patient too**,
so both eyes of a patient always land in the same fold. The screen measures use the same grouped
outer folds; their stacking step's inner folds are not grouped, which is fine because the
measures are descriptive only.

**Verdict:** each dataset gets exactly one, checked in this order:
1. **Classical enough:** the best classical or nonlinear student is within 0.01 of the exact
   map, or above it.
2. **Mean-field enough:** mean-field (or J=0, or ground state) is within 0.01 of the exact map.
3. **Exact ahead:** margin ≥ 0.05 over every other student. This makes the dataset a
   **candidate**, and it goes to confirmation.
4. **Exact slightly ahead:** 0.01 < margin < 0.05. Reported; no claim.

**Recorded for every dataset:**
- D_Q, the most common choice over the outer folds;
- D\* and R = D\*/D_Q;
- G, and the mean-field version of G;
- the full classical curve;
- the screen measures (L1, L2, F1), as a description.

## 7. Confirmation half

**Fitting.** For every student, the setting and regularisation are chosen by 5-fold
cross-validation on the **whole discovery half**. The student is refit on the discovery half and
scored **once** on the confirmation half. 95% intervals come from 1000 bootstrap resamples of
the confirmation half, resampling **patients** where there are groups.

**Claims we will make, and only these:**
- **Confirmed exact win:** confirmation margin ≥ 0.05 **and** the interval's lower end > 0.
  Only datasets that were candidates in the discovery half are eligible.
- **Confirmed map compactness:** G on the confirmation half is ≥ 0.02 and the interval's lower
  end is > 0, for the exact map **or** for mean-field.
  - Each is compared with the best classical expansion that has **at most its own feature
    count**. The count comes from its final fit.
  - That classical expansion is chosen by discovery cross-validation, never by looking at the
    confirmation half.
- **Confirmed quantum part:** exact − mean-field on the confirmation half has an interval lower
  end > 0. Only counted where map compactness is confirmed.
- **D\* on confirmation, descriptive only:** the discovery D* expansion, refit and scored on the
  confirmation half, set beside the exact map.

## 8. Predictions (written before any real data are modelled)

| | Prediction | Why we expect it |
|---|---|---|
| **R1** | Verdict "classical enough" on at least 5 of the datasets that are run | On non-AQFM resistant data the map helped 4% of the time (screen study, P10), and clinical tables are mostly simple |
| **R2** | **No confirmed exact win** on any dataset | Same reason |
| **R3** | Confirmed map compactness on **at most one** dataset | Raw linear features (D = p) are inside every equal-count comparison, and clinical data tend to be close to additive |
| **R4** | Wherever map compactness is confirmed, mean-field is as good as exact (no confirmed quantum part) | Matches the τ sweep everywhere outside the window |
| **R5** | The interaction ladder (F1) calls order 1, additive, on at least 5 of the datasets that are run | Clinical tabular data are mostly additive |

If R2 and R3 hold, the abstract still has a clean result: the ladder applied to every
public tabular ophthalmic dataset we could find, how many it stopped at each rung, and what
compactness the map does or doesn't offer. That's useful precisely because it's honest.

## 9. What gets reported, whatever happens

- One table row per dataset: n, positives, p, the verdict, the margin (discovery and
  confirmation, with an interval), D_Q, D\*, R, G with an interval, exact − mean-field with an
  interval, the reference all-features AUC, and F1 order.
- Every dataset, including those dropped before `run`, with the reason.
- **Caveats:**
  - small n, so the intervals will be wide;
  - only a handful of datasets;
  - using 8 chosen features can throw away signal the map might have used, and the
    all-features reference shows how much;
  - "compact" is not "cheap".

## 10. Amendments

None yet.
