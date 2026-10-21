# Can we tell in advance when a dataset needs AQFM?

*Short summary of the screen study, 2026-10-21. The full plan, with every check written down
before the runs, is in `screen_plan.md`. The code is `experiments/ts_screen.py`.*

## The short answer

No, not in the sense of "this dataset needs a quantum feature map." But we can do something
almost as useful:

> **We can cheaply tell you when you *don't* need a quantum feature map. When you might, we can
> tell you exactly what evidence justifies trying it. And on data the map didn't generate,
> expect it to help rarely.**

That's a decision procedure, not a predictor. It's weaker than "we can spot quantum-worthy
datasets in advance," but it's defensible, and the last sentence is a finding in its own right.

## What we did

We built a library of **1152 synthetic datasets** where we know the right answer:
- **864 made by the map itself** (the τ sweep from `aqfm_overview.tex`);
- **126 noisier copies** of some of them;
- **162 made by ordinary, non-quantum rules**: linear, pairwise and three-way interactions,
  parity-like rules, small random neural nets, and pure noise.

Every dataset got the full set of students: the exact map, its classical stand-ins (mean-field,
J=0, ground state), tuned classical feature maps, and strong nonlinear learners (boosted trees,
SVMs, neural nets). We then asked whether cheap tests on the data alone could predict where the
exact map wins.

Two terms used below:
- **Wins / helps:** the exact map beats every other student by at least 0.05 AUC.
- **Resistant:** the best cheap model falls at least 0.05 short of the best possible AUC.

## What we found

**1. The map helps almost only on data it generated.**

| Resistant datasets made by... | Map helps on |
|---|---|
| the map | **82%** (90 of 110) |
| ordinary rules | **4%** (3 of 70) |

On ordinary rules the map is usually *worse* than classical. On three-way interactions and
three-input parity it's about 0.3 AUC behind, because polynomials handle those easily and the
map doesn't.

**2. A classical test can't tell you a dataset needs the map.** This is a limit of principle, not
of our tests. If a classical test could detect signal that only the map reaches, that test
would itself be a classical model reaching it. The data agree:
- Our best data-only score ranked resistant datasets at 0.65 AUROC. We'd set 0.75 as the bar.
- The best ceiling estimate (from learning curves) saw only about half the real headroom on
  resistant datasets: 0.07 of a median 0.12.
- Parity over all inputs was invisible to every classical method.

**3. Telling you that you *don't* need it works well.**
- **If you know the ceiling,** "ceiling minus best cheap model" predicts where the map wins
  almost perfectly (0.976). The logic is sound; the hard part is knowing the ceiling.
- **The map's classical stand-in sorts out the look-alikes.** Data made by the mean-field
  version of the map looks *more* resistant to classical models than data made by the exact
  map. Running the mean-field simulation, which is cheap and classical, separates the two
  almost perfectly (0.96–0.99).

**4. Label noise erodes the map's advantage but doesn't erase it.** In the region where the
exact map wins, it wins on 83% of datasets with clean labels and 62% with much noisier ones.

## The decision procedure

1. **Is there anything left to find?** Run the cheap classical models and compare them with the
   ceiling. If they're already there, stop.
2. **Does the map's classical stand-in get it?** Run the mean-field simulation. If it closes the
   gap, stop: the data only *look* quantum-shaped.
3. **Pilot the exact map** on whatever survives.
   - Up to about 20 qubits, this is still a classical simulation, so it costs computer time, not
     quantum hardware.

Each step has a stop rule set in advance, so this isn't "keep trying until something works."
Reaching step 3 comes with an argument for it: the cheap models leave this much unexplained, and
the map's classical twin doesn't close the gap.

## Open threads

- **Making step 1 work on real data.** It needs a ceiling estimate that doesn't depend on
  classical learners. A promising route for biomedical data is **replicate labels**: two raters,
  a test–retest, or duplicate assays. How often two independent labels agree gives an upper bound
  on the best achievable accuracy. This is the "noise ceiling" idea from neuroscience
  (Nili et al. 2014).
  - On five of our datasets it came out 0.03–0.06 above the truth, as a bound should.
  - It saw the parity signal that every classical method missed.
  - A proper test on the whole library is the next step.
- **One lead worth checking.** On parity over all 6 inputs, the exact map got 0.57–0.63 where
  every classical model sat at chance: the only place it beat classical on data it didn't make.
  It isn't a result yet. The three datasets share the same rule and inputs, and the effect
  disappears with noisier labels. A small replication would settle it.
- **Pilot cost.** How small a pilot (fewer map settings, fewer samples) still gives the right
  call.

## Caveats

- Everything here is synthetic, at 4–8 qubits, with one set of inputs per size.
- "Ordinary rules" are nine families we chose. Real biomedical structure may differ.
- The map's classical stand-ins are cheap at these sizes. At larger sizes the ground-state
  stand-in gets expensive (it grows as 2^p), though mean-field doesn't.
