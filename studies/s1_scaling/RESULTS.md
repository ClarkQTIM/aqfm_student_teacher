# S1 results, Part A: how complicated are the map's label functions?

*Run 2026-10-27/28 on ophlappc06: exact banks on 2 GPUs, mean-field and J = 0 on CPU. The plan
(Part A predictions fixed in commit 045d6e7) is in `PLAN.md`. Log: `runs/s1_partA.log`. Table:
`runs/studies/s1/S1_PARTA.csv`. Figures are made from that table by `figures.py`.*

## In plain language

**What we asked.** In the window, simple classical models can't copy the quantum feature map.
One possible reason is that the map's labels are complicated functions of the inputs. We
measured how complicated, and whether that gets worse as we add qubits.

**How.** For 2 to 12 qubits, we ran the map on 40,000 random inputs and built label functions
("teachers") from its outputs. We then tried to reproduce each one with polynomials of the
inputs: first straight lines, then squares and pairwise products, then cubes, and so on up to
about 1,300 terms. We compared four kinds of teacher:
- the exact quantum map in the window;
- its cheap mean-field approximation;
- an uncoupled version (J = 0), where each output depends on one input only;
- an easy control with a short anneal.

**What we found.**

- **Outside the window, polynomials work.** For the short-anneal control, degree-2 polynomials
  (44 terms at 8 qubits) reproduce 94% of the label function, and degree 3 gets 98%.
- **In the window, they don't.** At 8 qubits, even 1,300 polynomial terms reproduce only about a
  quarter of the exact map's label function. The map's own 24 measured numbers reproduce all of
  it.

![Share of the label function reproduced against the number of polynomial features, p = 8](figures/fig_partA_efficiency_p8.png)

- **It doesn't get worse with more qubits.** The share polynomials miss stays between about 0.6
  and 0.8 from 2 qubits to 12. We had predicted it would grow; it didn't.

![Share of the label function cubic polynomials miss, against the number of qubits](figures/fig_partA_vs_qubits.png)

- **The quantum part isn't the cause.** Mean-field labels are at least as hard for polynomials,
  and uncoupled qubits are nearly as hard. What makes the difference is the **long anneal**: it
  makes each qubit's output swing back and forth rapidly as its input changes, and polynomials
  are bad at following that.

**What it means.** The difficulty polynomials have in the window is real and large, but it's set
by the anneal time, not by the number of qubits, and it isn't specifically quantum. The quantum
part shows up elsewhere: S2 found that mean-field can't reproduce the exact map in the window at
any size up to 16 qubits. The next question is whether classical models built for oscillating
functions (Fourier features, neural networks) also struggle, and whether *that* grows with
qubits. That's Part B. Polynomial degree turned out to be the wrong yardstick for these
functions.

**One caveat.** At very high polynomial degree with few qubits, the fitting became numerically
unreliable. That affects only one side table, not the results above, and it will be fixed
before that table is used.

---

## Details

### Checks (pre-registered; the test is at degree 3, degree 4 is reported)

A slope is the change per qubit in V_d, the share of the label function that polynomials of
degree ≤ d miss, fitted over p = 4–12. Each group's number is the median over its teachers;
the 95% intervals come from a bootstrap over teachers.

| | Prediction | Degree 3 | Degree 4 | |
|---|---|---|---|---|
| A1 | Exact, window: V_d grows with p (slope > 0) | −0.0097 [−0.0206, +0.0014] | −0.0047 [−0.0174, +0.0072] | **failed** |
| A2 | …faster than the exact control cells | −0.0051 [−0.0164, +0.0103] | −0.0025 [−0.0152, +0.0105] | failed |
| A2 | …faster than J = 0 at the window's anneal times | −0.0066 [−0.0247, +0.0146] | −0.0027 [−0.0239, +0.0161] | failed |
| A3 | Exact minus mean-field slope has an interval containing 0 | −0.0070 [−0.0217, +0.0092] | −0.0052 [−0.0232, +0.0095] | held, trivially (neither grows) |

**Outcome (F)** in the plan's terms: in polynomial degree, the label functions don't get more
complex with p.

### V₃ by group and p (median)

| | p = 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| exact, window | 0.83 | 0.79 | 0.74 | 0.80 | 0.69 | 0.75 | 0.68 | 0.60 |
| mean-field, window | 0.75 | 0.78 | 0.76 | 0.75 | 0.79 | 0.81 | 0.77 | 0.72 |
| J = 0, window anneal times | 0.52 | 0.51 | 0.67 | 0.60 | 0.60 | 0.58 | 0.63 | 0.60 |
| exact, control cells | 0.09 | 0.17 | 0.28 | 0.28 | 0.26 | 0.25 | 0.35 | 0.20 |
| J = 0, short anneal (τ = 1) | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

**Groups:**
- exact, window and mean-field, window: 36 teachers each (12 settings × 3 readouts);
- J = 0, window anneal times: 12 teachers (4 anneal times × 3 readouts);
- exact, control cells: 6 teachers (2 cells × 3 readouts);
- J = 0, short anneal: 3 teachers.

### At p = 8: share reproduced (test R²) against the number of polynomial terms

| terms (degree) | 8 (1) | 44 (2) | 164 (3) | 494 (4) | 1,286 (5) |
|---|---|---|---|---|---|
| exact, window | 0.21 | 0.23 | 0.25 | 0.27 | 0.25 |
| mean-field, window | 0.17 | 0.17 | 0.20 | 0.19 | 0.16 |
| J = 0, window anneal times | 0.33 | 0.36 | 0.42 | 0.45 | 0.49 |
| J = 0, short anneal | 0.67 | 0.94 | 0.98 | 1.00 | 1.00 |

**Why some curves dip.** The small dips at 4–5 are the expected out-of-sample cost of adding
terms that capture nothing new. A fit with m terms on 30,000 rows inflates the missed share by
about 1/(1 − m/30,000), which is 4% at 1,286 terms.

### Known problem: high-degree fits at small p

For p ≤ 5 at degree 8 and above, V_d *rises* with degree, up to impossible values (V > 1). With
45–1,800 terms on 30,000 rows that can't be overfitting. It's the solver: it squares the
condition number of the high-degree Hermite design.

**What it affects.** The "degree needed for R² ≥ 0.90 / 0.99" tables in the log, at p ≤ 5. Do not
quote those.

**What it doesn't affect.** Anything at degree ≤ 5: every V there is ≤ about 1.0, with no
blow-ups. That includes the degree-3 test, the degree-4 robustness check, and both figures.

**The fix,** for a later amendment: solve by QR on the design, not the normal equations.

### Interpretation, and what's next

The "long anneal → oscillation" reading comes from the J = 0 contrast. Uncoupled qubits are
easy for polynomials at τ = 1 (V₃ = 0.02) and hard at τ = 3–20 (0.5–0.67), at every p. It's an
interpretation, not a separate test.

**Proposed next measurement** (an amendment, post hoc): **interaction order** instead of
degree. How much of each label function is a sum of one-input functions, however wiggly? How
much needs pairs of inputs, and how much more than pairs? That's the part that could grow with
p, and the part the coupling should affect. Part B (Fourier, random features, neural networks,
boosted trees, by size and by sample count) gets its predictions fixed after that.
