# Geometric difference: results

*Run 2026-10-22 on the server, following `docs/geometry_plan.md`, which was written before the
run. Log: `runs/geometry.log`. Tables: `runs/tau_sweep/geometry/GEO_TT.csv` (864 teacher
datasets) and `runs/GEO_REAL.csv` (59 real datasets). The run took about 3 minutes on 31
cores.*

## In one paragraph

Huang et al. (2021) define the geometric difference g. It measures how far a copier's kernel
is from a target's kernel, *without looking at labels*. A small g means no labels on those
samples can favour the target by much.

**In the τ sweep, g lines up with our window almost perfectly.** All five predictions held:
- g between the map's kernel and Gaussian kernels on the inputs picks out the datasets where
  classical opponents fall short (AUROC 0.94).
- g between the exact map and its own mean-field version picks out the datasets where exact
  beats mean-field (AUROC 0.95).

So the window isn't an artefact of our opponents: it's where the geometry says the map
differs. It also explains why entanglement isn't enough. Short, strongly coupled anneals are
*more* entangled than the window, but their g is small.

**On real data, both predictions failed, and both tests were badly designed.**
- The rule-out threshold was calibrated at N = 600, and it mostly flags small datasets.
- The label-aware ratio is dominated by datasets with no signal.

What the real data do show is consistent with Huang's framing. Many real datasets have
inputs on which the map *could* differ (large g), but their labels don't use that room.

## 1. τ sweep: the predictions

At the primary λ = 0.01, exact teachers, full readout, 180 datasets:

| | Prediction | Result | |
|---|---|---|---|
| G1 | g_classical predicts "classical short by ≥ 0.05": AUROC ≥ 0.75, Spearman ≥ 0.5 | AUROC **0.94** (113/180 short), Spearman **0.72** | held |
| G2 | g(MF ‖ Q) predicts "exact − mean-field ≥ 0.05": AUROC ≥ 0.75, Spearman ≥ 0.5 | AUROC **0.95** (81/180 ahead), Spearman **0.87** | held |
| G3 | Median g(MF ‖ Q): window ≥ 2× the weak-coupling / short-anneal region | **3.6×** | held |
| G4 | Z-only readout gives a lower g than full readout in ≥ 80% of window cells | **83%** of 63 cells | held |
| G5 | Mean-field teachers' g_classical in the window ≥ 0.5× exact teachers' | **1.24×** | held |

**Median g by region** (exact teachers; best Gaussian width for each copier):

| readout | region | g_classical | g(MF ‖ Q) | g(J0 ‖ Q) | classical shortfall | exact − MF |
|---|---|---|---|---|---|---|
| full | weak coupling or short anneal | 6.8 | 3.9 | 5.0 | 0.013 | 0.001 |
| full | in between (c = 1) | 13.9 | 6.0 | 13.3 | 0.142 | 0.031 |
| full | other (edges) | 10.0 | 8.4 | 7.8 | 0.044 | 0.042 |
| full | **window** | **16.2** | **13.9** | **14.3** | **0.155** | **0.187** |
| Z-only | all | 8.4 | 6.2 | 12.7 | 0.003 | 0.000 |

Mean-field teachers show the same pattern, with g_classical 7.3 below the window and 20.0 in
it. That's the result "mean-field labels are classically hard too" in Huang's terms. A large
g against classical kernels comes from the Ising-type map itself, not from anything quantum.
The twin comparison, g(MF ‖ Q), is the one that isolates the quantum part.

**Sensitivity to λ (not part of the test).**
- **λ = 0.001:** every check holds.
- **λ = 0.1:** everything holds except G1's Spearman (0.47) and G4 (27%).
- **λ = 1:** almost everything fails. At that level the ridge washes out every kernel's fine
  structure, so only the coarsest directions count. The result depends on not
  over-regularizing, and any write-up should say so.

## 2. τ sweep: what we saw beyond the predictions (exploratory)

- **g explains why entanglement isn't enough.**
  - Strongly coupled short anneals (c ≥ 2, τ ≤ 1) have median entanglement 0.88, above the
    window's 0.80. Yet their g is 5.3 (classical) and 4.4 (twin), and exact − mean-field
    ≈ 0.
  - Across exact teachers, the Spearman correlation with exact − mean-field is **0.42 for
    entanglement and 0.87 for g(MF ‖ Q)**.
  - The twin's **linear** kernel tells the same story through its training error t (the part
    of the exact map's features that mean-field features can't express at all): Spearman
    0.82 with exact − mean-field, AUROC 0.91.
- **Inside the window, g hardly ranks datasets** (AUROC 0.67 classical, 0.59 twin). That's
  expected: 58 of 63 window datasets are classically short and 59 have exact ahead of
  mean-field, so there's little left to rank. Outside the window, g still ranks well: AUROC
  0.75–1.0 in the in-between and edge regions.
- **In the window, g shrinks with p, but the measured gap doesn't:**

  | p | g_classical | g(MF ‖ Q) | exact − MF | classical shortfall |
  |---|---|---|---|---|
  | 4 | 20.1 | 18.6 | 0.19 | 0.16 |
  | 6 | 16.4 | 14.2 | 0.14 | 0.15 |
  | 8 | 13.0 | 11.4 | 0.24 | 0.18 |

  Outside the window, g(MF ‖ Q) also falls with p (5.3 → 4.3 → 3.9). That's what "mean-field
  becomes more accurate as p grows" would look like with all-to-all coupling. However, g
  depends on the kernel's dimension at fixed N, so the trend may not be physical. The two
  measures point in different directions, which is exactly what the p = 10–12 run would
  settle.
- **g² (130–400 in the window) is a worst-case cost over all labels**, not a forecast for these
  labels. The bound s_C ≤ g² s_T is often loose. A sample-size sweep should therefore predict
  catch-up from a label-aware quantity, not from g².

## 3. Real data

| | Prediction | Result | |
|---|---|---|---|
| G7 | Datasets whose largest g_classical (over 60 settings) is below g₀ = 11.8 are all "classical enough", with no confirmed lead | 13 of 59 below g₀. Two exceptions: **grape** ("exact slightly ahead" on discovery) and **biomed** ("mean-field enough"). Neither held on confirmation | failed |
| G8 | monk2 among the top 3 pool datasets by s(MF)/s(exact) | rank **17** of 54 | failed |
| G9 | Share below g₀ (descriptive) | 22% | |

**Why G7 is uninformative as designed.**
- **Sample size.** All 13 datasets below g₀ have ≤ 303 rows, and every dataset with 600 rows
  is above it. g grows with N, and g₀ was calibrated at N = 600, so "below g₀" mostly means
  "small dataset". The plan flagged this risk, and the test ran into it anyway.
- **Unequal comparison.** Real datasets were scored by their *maximum* over 60 settings; the
  τ-sweep values come from one setting each. Taking the maximum pushes real g upward.

The two exceptions are discovery-half verdicts that collapsed on confirmation
(lower interval ends −0.23 and −0.06), so neither is a real counterexample. Still, the
prediction as written failed.

**Why G8's measure is broken.** The ratio s(MF)/s(exact) rewards datasets where the map's
linear kernel fits almost nothing. There s(exact) is tiny and the ratio blows up. The top of
the list is dmft, breast_cancer and cloud, all without signal. The complexity s should have
been paired with training error, as g is paired with t.

**What the real data do show.**
- At its best setting, the map's kernel is at least as far from classical as the window's
  median (g ≥ 16.2) on **30 of 59** real datasets. So on the inputs alone, there is often room
  for the map to differ.
- Nothing wins anyway. In Huang's terms, a large g is necessary, not sufficient, and these
  labels don't use the room.
- At the settings the exact student actually chose, the median g_classical is **6.9**. That's
  the level of the τ sweep's weak / short region, where classical always catches up.

## 4. What this means

- **For the teacher–student paper.** The window now has a theory-backed description that
  doesn't depend on our choice of opponents. That answers the objection a reviewer would
  raise first, and it links us directly to Huang et al.
- **For scaling.** g already hints that the twin gap might close at larger p. The p = 10–12
  run decides it.
- **For real data.** Our nulls are "the labels don't use the room", not "there's no room". That
  points the search towards datasets whose labels depend on collective structure.
- **Possible fixes, not run (would be post-hoc amendments):**
  - G7: recalibrated per dataset, by subsampling the τ sweep to the dataset's N and taking the
    same maximum over settings.
  - G8: replaced by a label-aware measure that accounts for training error, such as
    cross-validated kernel-ridge error for each kernel.
