# S3: Other couplings: is there a window that doesn't close?

*Plan, PARKED 2026-10-26. To be finalized after S2, whose result decides how urgent this is.*

## 1. The question

Mean-field becomes exact as p grows *because* of our coupling, J = c/(p−1) between every
pair: each qubit feels the average of many weak pulls. Other couplings have no such limit.
Is there a coupling where the window survives as p grows, and where *no cheap classical
method* closes it, not just plain mean-field?

## 2. Couplings

Each qubit's total coupling strength is kept at c, as now, so the cells stay comparable.

| Coupling | J_ij | Why it's interesting |
|---|---|---|
| All-to-all (current) | c/(p−1) for every pair | The baseline; mean-field becomes exact as p grows |
| Nearest-neighbour ring | c/2 for neighbours i, i±1; 0 otherwise | A qubit feels 2 neighbours at full strength, so mean-field never becomes exact |
| Random-sign all-to-all | c·s_ij/√(p−1), with s_ij = ±1 at random (fixed per p) | Mean-field misses a correction that doesn't shrink with p |

The simulators already accept a coupling matrix (`compute_bank` and `meanfield_local`), so
the exact and mean-field sides need no new code.

## 3. The catch: "mean-field fails" is not "classical fails"

The fair twin for each coupling is the best *cheap* classical method, not plain mean-field:

- **Ring:** information spreads only a limited distance during the anneal. So each qubit's
  readout depends mostly on nearby inputs, and the right twins are:
  - local classical features: products of neighbouring inputs;
  - a tensor-network simulation, which is cheap for short-time 1D dynamics.
- **Random signs:** mean-field with the standard correction for random-sign couplings (the
  Onsager reaction term) is the right twin.

## 4. What we expect

- **Ring:** mean-field stays wrong as p grows, but local classical features and tensor networks
  catch up. So no lasting window.
- **Random signs:** both corrected mean-field and generic classical learners stay behind as p
  grows. **This would be the most interesting outcome.**

## 5. Design (sketch)

- Repeat S2 (mean-field error vs p) for each coupling, with the right twin added.
- Repeat a light version of S1: Part A (polynomial degree needed), plus the best two classical
  families from S1.
- Same cells as S1/S2, possibly re-centred if the window moves under the new coupling. A short
  τ × c scan at p = 6 comes first to find out.

## 6. Open decisions

- Whether to implement the corrected mean-field and a tensor-network twin, or start with
  plain mean-field plus local classical features.
- Whether the random-sign couplings get several random draws per p, which seems needed so a
  single lucky draw doesn't decide the result.
