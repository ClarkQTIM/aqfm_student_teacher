# Real-data results: the ladder on five tabular ophthalmic datasets (2026-10-21)

**Plan:** `realdata_plan.md`, with Amendments 1–2. Everything below was fixed before the run,
except the PAPILA and GRAPE diagnostics, which are marked *exploratory*.

**Run:** `runs/realdata`, about 26 minutes on 32 cores, with no failures.

- **Five datasets run:** DR Debrecen, PAPILA, myopia (Orinda), GRAPE and FIVES/VascX.
- **Not run:** keratoconus (no file obtained) and PERG-IOBA (needs signal processing).
- **Setup for each dataset:** 8 qubits chosen by random-forest importance on the discovery half;
  the τ-sweep students; nested cross-validation on the discovery half; one scoring on the held-back
  confirmation half.

## AUC by student (discovery → confirmation)

| Dataset | n (positive) | Exact map | Mean-field | J=0 | Best classical | Nonlinear (8 inputs) | Nonlinear (all features) | Verdict |
|---|---|---|---|---|---|---|---|---|
| DR Debrecen | 1151 (611) | 0.78 → 0.80 | 0.79 → 0.80 | 0.79 → 0.80 | 0.78 → 0.80 | 0.75 → 0.74 | 0.79 → 0.81 | classical enough |
| Myopia | 618 (81) | 0.81 → 0.90 | 0.81 → 0.90 | 0.81 → 0.90 | 0.80 → 0.89 | 0.74 → 0.75 | 0.74 → 0.81 | classical enough |
| FIVES/VascX | 800 (600) | 0.81 → 0.83 | 0.81 → 0.83 | 0.80 → 0.83 | 0.81 → 0.83 | 0.80 → 0.83 | 0.79 → 0.77 | classical enough |
| PAPILA | 420 eyes / 47 glaucoma patients | 0.79 → 0.57 | 0.85 → 0.58 | 0.80 → 0.70 | 0.81 → 0.74 | 0.80 → 0.65 | 0.75 → 0.66 | classical enough |
| GRAPE | 263 eyes (40 progressing) | 0.53 → 0.53 | 0.43 → 0.50 | 0.43 → 0.45 | 0.51 → 0.57 | 0.48 → 0.52 | 0.46 → 0.57 | "exact slightly ahead" (noise: see below) |

## Feature count: no compactness

| Dataset | Map features used (D_Q) | Classical features needed to match (D\*) | R = D\*/D_Q | Equal-count gap G, confirmation (interval's lower end) |
|---|---|---|---|---|
| DR Debrecen | 24 | 8 | 0.33 | +0.008 (−0.012) |
| PAPILA | 24 | 8 | 0.33 | −0.071 (−0.137) |
| Myopia | 8 | 8 | 1.0 | +0.001 (−0.001) |
| FIVES/VascX | 8 | 8 | 1.0 | −0.006 (−0.029) |
| GRAPE | 28 | 8 | 0.29 | (no signal) |

Classical expansions match the map with **as few or fewer** features on every dataset.
Compactness would need G ≥ 0.02 with a lower end above 0, and no dataset comes close.

## Checks (predictions written before the run)

| | Prediction | Outcome |
|---|---|---|
| R1 | "classical enough" on all datasets run | **4 of 5.** The fifth is GRAPE, where nothing has signal |
| R2 | no confirmed exact win | **holds** (0 of 5) |
| R3 | confirmed map compactness on at most one | **holds** (0 of 5) |
| R4 | no confirmed quantum part | **holds** (nothing to test) |
| R5 | interaction ladder says additive on all | **3 of 5.** DR reads order ≥ 4; GRAPE "none found" |

## What it says

1. **The map doesn't help on any of these datasets.** On the three with clear signal (DR,
   myopia, FIVES), every student reaches the same AUC on the confirmation half, within ±0.003.
2. **The map isn't more compact either.** Classical expansions match it with 8 features or fewer,
   where the map uses 8–24.
3. **The coupling adds nothing.** The J=0 student, which has no coupling between qubits, matches
   the exact map everywhere. Its features are a nonlinear transform of each input *separately*,
   so it amounts to an additive model. These clinical tables don't contain the kind of joint
   structure the map's coupling would add, which fits the interaction ladder reading three of them
   as additive.
4. **This is what the screen study predicted.** On non-AQFM data the map helped about 4% of the
   time, and on real ophthalmic tables it helped 0 of 5 times.

## Diagnostics (exploratory, done after seeing the results)

- **PAPILA's discovery-to-confirmation drop is in the data, not the code.**
  - Plain linear regression on the same 8 inputs, which doesn't use the map code at all:

    | | AUC |
    |---|---|
    | cross-validation within discovery | 0.84 |
    | cross-validation within confirmation | 0.58 |
    | trained on discovery, tested on confirmation | 0.64 |
    | trained on confirmation, tested on discovery | 0.81 |

  - The single-feature signal differs between halves. Age goes from 0.83 to 0.68; corneal
    thickness from 0.64 to 0.44, reversing direction; IOP, axial length and lens status from about
    0.6 to about 0.5.
  - The 87 glaucoma eyes come from **47 patients**, about 24 per half, and most contribute both
    eyes. So the effective sample is small, and the halves can differ by chance.
  - The grouping is correct: no patient appears in both halves.
- **GRAPE has no signal from baseline.** Linear regression gets 0.46–0.50 within each half and
  across. The "exact slightly ahead" verdict is 0.53 against 0.51, which is noise.
  - Below-0.5 cross-validated AUCs are a known artefact when there is nothing to learn and few
    positives.
  - Lesson for future plans: include a "no signal" verdict, e.g. every student < 0.6.

## Caveats

- **Only five datasets,** two of them small in effective size: PAPILA (47 glaucoma patients) and
  GRAPE (35 progressing patients).
- **Random-forest importance favours continuous features over binary ones** (Strobl et al. 2007).
  - Parental myopia, a known risk factor, wasn't among myopia's 8 qubits.
  - The all-features reference learner did best on myopia's discovery half (boosted trees 0.83),
    but not on its confirmation half (0.81 vs 0.90 for the 8-input students).
- **FIVES has no patient IDs,** so images from the same person could fall in both halves.
- **"Compact" isn't "cheap":** each map feature would need about 1000 shots on hardware.
