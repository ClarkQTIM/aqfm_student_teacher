"""
Adapters for the real tabular datasets (docs/realdata_plan.md, Section 3).

Each adapter reads the raw files from  data/ophtho/<name>/  (inside the repo; git
ignores data/) and returns a Raw record:
    X       DataFrame of candidate features (dropped columns already removed)
    y       0/1 labels (1 = positive, as in the plan's table)
    groups  patient ids, or None when every row is its own patient
    info    dict: source, label definition, what was dropped and why

Two routes:
  * a dataset-specific adapter (below), written once the raw file's columns are known;
  * the GENERIC route for anything else: put  data/ophtho/<name>/clean.csv  with a
    column `label` (0/1), optionally a column `group`, and every other column a
    candidate feature.  `load(name)` falls back to this when no adapter exists.

Adapters only select and recode columns; all modelling choices (missing values,
splitting, picking the qubits, scaling) happen in realdata.py, identically for all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "ophtho"


@dataclass
class Raw:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray | None
    info: dict = field(default_factory=dict)


def _root(root):
    """the data folder, looked up at CALL time (so it can be redirected, e.g. in tests)"""
    return DATA_ROOT if root is None else Path(root)


class NotReady(RuntimeError):
    """the raw file isn't there yet, or the adapter still needs finishing"""


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise NotReady(f"missing {path}\n  {hint}")
    return path


# ---------------------------------------------------------------------------
# generic route
# ---------------------------------------------------------------------------

def load_generic(name, root=None):
    root = _root(root)
    f = _need(root / name / "clean.csv",
              "generic route: a CSV with column `label` (0/1), optional `group`, other columns = features")
    D = pd.read_csv(f)
    if "label" not in D:
        raise NotReady(f"{f} has no `label` column")
    y = D.pop("label").to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        raise NotReady(f"{f}: `label` must be 0/1, found {sorted(np.unique(y))[:5]}")
    groups = D.pop("group").to_numpy() if "group" in D else None
    return Raw(D, y.astype(int), groups, {"source": str(f), "route": "generic clean.csv"})


# ---------------------------------------------------------------------------
# dataset-specific adapters
# ---------------------------------------------------------------------------

DEBRECEN_NAMES = (["quality", "prescreen"] + [f"ma_{a}" for a in ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0")]
                  + [f"exudate_{i}" for i in range(1, 9)] + ["macula_od_dist", "od_diameter", "amfm"])


def _read_arff(path):
    """minimal ARFF reader: numeric attributes, comma-separated @data rows"""
    names, rows, in_data = [], [], False
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            names.append(s.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([v.strip() for v in s.split(",")])
    return pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")


def load_dr_debrecen(root=None):
    """UCI 329 (CC BY 4.0): 1151 Messidor images, 19 features, label = DR signs.
    File: messidor_features.arff from the UCI zip."""
    d = _root(root) / "dr_debrecen"
    f = _need(d / "messidor_features.arff",
              "download the zip from https://archive.ics.uci.edu/dataset/329 and unzip it here")
    D = _read_arff(f)
    if D.shape[1] != 20:
        raise NotReady(f"{f}: expected 20 columns (19 features + class), found {D.shape[1]}")
    y = D.iloc[:, -1].astype(int).to_numpy()
    X = D.iloc[:, :-1].copy()
    X.columns = DEBRECEN_NAMES
    return Raw(X, y, None, {"source": str(f), "label": "DR signs (Messidor 1-3) vs none", "dropped": []})


def load_myopia(root=None):
    """Orinda Longitudinal Study (R package aplore3).  File: myopia.csv exported with
       R -e 'install.packages("aplore3"); write.csv(aplore3::myopia, "myopia.csv", row.names=FALSE)'"""
    d = _root(root) / "myopia"
    f = _need(d / "myopia.csv", "export it from R with the command in this adapter's docstring")
    D = pd.read_csv(f)
    # R factors export either as labels ("No"/"Yes", "Male"/"Female") or as codes (1/2)
    def code(v, positive):
        return v.astype(str).str.strip().str.lower().isin(positive).astype(int)
    y = code(D["myopic"], {"yes", "2", "2.0"}).to_numpy()
    X = D.drop(columns=["id", "studyyear", "myopic"])
    for c in ("mommy", "dadmy"):
        X[c] = code(X[c], {"yes", "2", "2.0"})
    X["gender"] = code(X["gender"], {"female", "2", "2.0"})
    return Raw(X, y, None, {"source": str(f), "label": "myopic within 5 years",
                            "dropped": ["id", "studyyear (administrative)"]})


def _todo(name, what):
    def f(root=None):
        root = _root(root)
        if (root / name / "clean.csv").exists():
            return load_generic(name, root)
        raise NotReady(f"adapter for {name} not finished: {what}\n"
                       f"  (or provide data/ophtho/{name}/clean.csv for the generic route)")
    return f


ADAPTERS = {
    "dr_debrecen": load_dr_debrecen,
    "myopia": load_myopia,
    "papila": _todo("papila", "needs the clinical-data columns; drop mean defect, glaucoma vs healthy, group = patient"),
    "grape": _todo("grape", "needs the baseline-sheet columns; label PLR2, drop other labels/follow-ups, group = patient"),
    "keratoconus": _todo("keratoconus", "needs the file's columns and class coding"),
    "perg_ioba": _todo("perg_ioba", "needs P50/N95 features from the signals, first visit per subject"),
    "fives_vascx": _todo("fives_vascx", "needs the file and its label column"),
}


def load(name, root=None) -> Raw:
    root = _root(root)
    if name in ADAPTERS:
        return ADAPTERS[name](root=root)
    return load_generic(name, root)


def available(root=None):
    """names of datasets whose files load right now"""
    root = _root(root)
    names = set(ADAPTERS) | ({p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set())
    ok = []
    for n in sorted(names):
        try:
            load(n, root)
            ok.append(n)
        except NotReady:
            pass
    return ok
