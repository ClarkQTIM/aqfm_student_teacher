# S4: Does a variational quantum circuit have the right inductive bias?

*Plan, PARKED 2026-10-26. Needs new code (a trainable circuit simulator); comes last.*

## 1. The question

In the window, classical models need much more than the map's 3p numbers to imitate it.
Does a **trainable quantum circuit with few parameters** reach the map's performance?

We're looking for **parity**: the circuit matches the exact student's AUC with far fewer
parameters than the classical models that get there (S1's size\*). If it does, its inductive
bias fits these labels.

## 2. Students, from generic to matched

| | Student | What it tests |
|---|---|---|
| (a) | **Generic circuit:** inputs as rotation angles, then trainable layers of rotations and entangling gates. Two versions: inputs encoded once, and inputs re-encoded every layer | Does *a* quantum model help? |
| (b) | **Same ingredients as the map, trainable schedule:** layers alternating the inputs as Z fields plus ZZ couplings, and an X driver, with trainable strengths and times | Does a quantum model with the *right physics* help, without being told the exact setting? |
| (c) | **The exact map at the teacher's setting** (the open-book student; exists already) | The ceiling |

All students read out ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit, followed by logistic regression, the same
readout as the map.

## 3. What we expect

- **(a) does about as well as classical models, not better.**
  - With inputs encoded once, a circuit can only make low-frequency functions of each input
    (Schuld et al. 2021), and the window labels oscillate.
  - Re-encoding helps, but such models are often matched by random Fourier features (Landman
    et al. 2022; Sweke et al. 2025).
- **(b) reaches near parity with a few dozen parameters.** That would say the advantage comes
  from the *right physics*, not from "being quantum".

Either outcome is useful. The one that would change the story is (a) reaching parity.

## 4. Design (sketch)

- p ∈ {4, 6, 8}, the 12 window cells, and the same datasets as S1 (n up to 3,000 for
  training).
- Parameter counts between 10 and about 500, so they can be set against S1's classical
  size\*.
- Simulation by statevector with automatic gradients (PyTorch), on GPU if available.
- **Open:** circuit depth and layer counts, optimizer and training budget. Also how to keep the
  comparison fair: same tuning budget and validation split as the classical models.
