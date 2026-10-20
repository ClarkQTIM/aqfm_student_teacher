# aqfm_student_teacher

When does an annealed quantum feature map produce labels that nothing classical can copy?
Teacher–student experiments: the quantum map (or one of its classical versions) makes the labels, and a set of
students, including classical and nonlinear learners, try to learn them.

The write-up is `docs/aqfm_overview.tex`. Every number in it can be regenerated and checked with the tools below.

## Layout

```
aqfm_prescreen/          the map and shared helpers (package name kept from the original repo)
  aqfm.py                exact / mean-field / J=0 simulators of the anneal, measurement banks
  dial.py                the 22 tuned classical feature maps, input standardization
  dictionaries.py        Fourier, Hermite and label-matched Fourier features
  evaluate.py            constants and fold helpers used by dial.py
  backend.py             numpy / optional GPU switch
experiments/
  teacher_student.py     students, nested CV, tuned logistic regression, classical curves; also the earlier 216-dataset run
  ts_tau_sweep.py        main run: 864 datasets (overview Sections 2-4)
  ts_addons.py           nonlinear opponents and shot noise on an existing run (Sections 4-5)
  ts_scaling.py          scaling to p = 12 (pending)
tools/
  overview_numbers.py    recomputes every number in the overview and checks it against the document
  verify_reproduction.py regenerates simulations, labels and score tables and compares them to the saved run
tests/                   unit tests (pytest), incl. a check of the simulator against the original runner
docs/aqfm_overview.tex   the write-up
runs/                    results of record (NOT in git; large)
  tau_sweep/             main run + add-ons  -> Sections 2-5
  teacher_student_v2/    earlier 216-dataset run (cited for counts only)
```

## House rules

Nothing is ever written to the home directory. Before running anything:

```bash
export PIP_NO_CACHE_DIR=1 TMPDIR=/tmp
```

The package also redirects library caches into the repo folder.

## Reproducing the overview

From the repo root:

```bash
export TT=$PWD/runs/tau_sweep

# 1. Main run (Sections 2-4): build inputs, simulations and labels, then run every student
python experiments/ts_tau_sweep.py make  --out $TT --workers 32
python experiments/ts_tau_sweep.py run   --out $TT --workers 32
python experiments/ts_tau_sweep.py score --out $TT > $TT/score.txt

# 2. Add-ons (Sections 4-5): nonlinear learners and shot noise, then the combined table
python experiments/ts_addons.py nonlinear --out $TT --workers 32
python experiments/ts_addons.py shots     --out $TT --workers 32
python experiments/ts_addons.py score     --out $TT > $TT/addons_score.txt

# 3. Check every number in docs/aqfm_overview.tex against the results
python tools/overview_numbers.py --runs runs
```

`run` and the add-on stages are restartable: finished datasets are skipped. After a crash, delete
`$TT/runs/*.tmp.csv` and rerun the same command.

## Verifying the saved results without re-running

```bash
python -m pytest -q tests
python tools/verify_reproduction.py --run runs/tau_sweep          # simulations, labels, score tables (~minutes)
python tools/verify_reproduction.py --run runs/tau_sweep --full   # + one complete student run (~5 more minutes)
python tools/overview_numbers.py --runs runs                      # all 47 quoted numbers
```

`verify_reproduction.py` writes only to `_verify/` inside the repo and never modifies `runs/`.
It prints the Python and library versions used; save its output next to the run
(e.g. `runs/tau_sweep/VERIFY_<date>.txt`).

## Status

- Sections 1-5 of the overview: complete, reproducible from `runs/tau_sweep`.
- Scaling run (`ts_scaling.py`, p = 4-12, 625 datasets): built and tested, not yet run.
