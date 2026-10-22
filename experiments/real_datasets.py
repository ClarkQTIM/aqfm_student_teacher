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


def _excel(path, sheet=0):
    try:
        return pd.read_excel(path, sheet_name=sheet, header=None)
    except ImportError as e:                                   # openpyxl missing
        raise NotReady(f"reading {path} needs openpyxl: PIP_NO_CACHE_DIR=1 pip install openpyxl ({e})")


def _check(cond, msg):
    if not cond:
        raise NotReady(msg)


PAPILA_COLS = ["id", "age", "gender", "diagnosis", "dioptre_1", "dioptre_2", "astigmatism", "phakic",
               "iop_pneumatic", "iop_perkins", "pachymetry", "axial_length", "vf_md"]
PAPILA_COUNTS = {0: 333, 1: 87, 2: 68}        # healthy, glaucoma, suspect (eyes; Kovalyk et al. 2022)
GRAPE_EYES, GRAPE_PLR2 = 263, 40              # Huang et al. 2023


def load_papila(root=None):
    """PAPILA (figshare 14798004): patient_data_od.xlsx + patient_data_os.xlsx, one row per
    patient per file.  Layout: row 0 = headers, row 1 = sub-headers (dioptre_1, dioptre_2,
    astigmatism; IOP Pneumatic / Perkins), row 2 = the word 'ID', data from row 3.
    Label (plan Section 3): glaucoma (1) vs healthy (0); suspects dropped.
    Dropped: VF_MD (recorded only for glaucoma patients: leaks the label), ID, eye side."""
    d = _root(root) / "papila"
    parts = []
    for eye in ("od", "os"):
        f = _need(d / f"patient_data_{eye}.xlsx", "download the clinical-data files from the PAPILA figshare")
        R = _excel(f)
        _check(R.shape[1] == 13 and str(R.iloc[0, 1]).strip() == "Age" and str(R.iloc[1, 4]).strip() == "dioptre_1",
               f"{f}: layout differs from the one this adapter was written for (shape {R.shape})")
        D = R.iloc[3:].copy()
        D.columns = PAPILA_COLS
        D = D[D.id.notna()]
        D["eye"] = eye
        parts.append(D)
    D = pd.concat(parts, ignore_index=True)
    for c in PAPILA_COLS[1:]:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    counts = D.diagnosis.value_counts().to_dict()
    _check({int(k): v for k, v in counts.items()} == PAPILA_COUNTS,
           f"PAPILA diagnosis counts {counts} differ from the paper's {PAPILA_COUNTS}: check the coding")
    D = D[D.diagnosis.isin([0, 1])]                            # suspects (2) dropped
    y = (D.diagnosis == 1).astype(int).to_numpy()
    groups = D.id.astype(str).str.strip().to_numpy()
    D = pd.concat([D, power_vectors(D.dioptre_1, D.dioptre_2, D.astigmatism)], axis=1)
    X = D.drop(columns=["id", "diagnosis", "vf_md", "eye", "dioptre_1", "dioptre_2", "astigmatism"]).reset_index(drop=True)
    return Raw(X, y, groups, {"source": str(d), "label": "glaucoma (1) vs healthy (0); suspects dropped",
                              "dropped": ["vf_md (leaks the label)", "id", "eye side", "suspect eyes"],
                              "transformed": "sphere, cylinder, axis -> power vectors M, J0, J45 (Amendment 2)"})


def power_vectors(sphere, cyl, axis_deg):
    """Refraction as power vectors (Thibos, Wheeler & Horner 1997): M = S + C/2,
    J0 = -(C/2) cos 2a, J45 = -(C/2) sin 2a.  Unlike the raw axis (where 0 and 180
    degrees are the same astigmatism), these behave as ordinary numbers."""
    a = np.deg2rad(pd.to_numeric(axis_deg, errors="coerce"))
    S, C = pd.to_numeric(sphere, errors="coerce"), pd.to_numeric(cyl, errors="coerce")
    return pd.DataFrame({"refr_M": S + C / 2, "refr_J0": -(C / 2) * np.cos(2 * a),
                         "refr_J45": -(C / 2) * np.sin(2 * a)}, index=sphere.index)


def load_grape(root=None):
    """GRAPE (figshare collection 6406319): 'VF and clinical information.xlsx', sheet
    Baseline.  Layout: row 0 = headers, row 1 = sub-headers (PLR2 / PLR3 / MD under
    'Progression Status'; Mean / S / N / I / T under 'OCT RNFL thickness'; 0..60 under
    'VF'), data from row 2.  Label (plan): progression by PLR2.  Dropped: PLR3 and MD-slope
    labels, Total Visits (a follow-up quantity, not known at baseline), image references.
    VF points coded -1 (not tested) become missing."""
    d = _root(root) / "grape"
    f = _need(d / "VF and clinical information.xlsx", "download 'GRAPE dataset: VF and clinical information.xlsx'")
    R = _excel(f, "Baseline")
    sub = R.iloc[1]
    _check(R.shape[1] == 80 and str(sub.iloc[7]).strip() == "PLR2" and str(sub.iloc[11]).strip() == "Mean"
           and str(sub.iloc[19]).strip() in ("0", "0.0"),
           f"{f}: layout differs from the one this adapter was written for (shape {R.shape})")
    D = R.iloc[2:].reset_index(drop=True)
    D = D[D.iloc[:, 1].notna()]                                  # a row per eye (laterality given)
    subject = D.iloc[:, 0].ffill()                               # in case of merged cells
    X = pd.DataFrame({"age": D.iloc[:, 2], "gender": D.iloc[:, 3], "iop": D.iloc[:, 4], "cct": D.iloc[:, 5],
                      "glaucoma_category": D.iloc[:, 10]})
    for i, nm in enumerate(["rnfl_mean", "rnfl_s", "rnfl_n", "rnfl_i", "rnfl_t"]):
        X[nm] = D.iloc[:, 11 + i]
    for k in range(61):
        X[f"vf_{k}"] = pd.to_numeric(D.iloc[:, 19 + k], errors="coerce").replace(-1, np.nan)
    for c in ("age", "iop", "cct", "rnfl_mean", "rnfl_s", "rnfl_n", "rnfl_i", "rnfl_t"):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(D.iloc[:, 7], errors="coerce")
    _check(len(X) == GRAPE_EYES and y.notna().all() and int(y.sum()) == GRAPE_PLR2,
           f"GRAPE: expected {GRAPE_EYES} eyes with {GRAPE_PLR2} PLR2-progressing, found {len(X)} eyes, {y.sum()} positive")
    return Raw(X.reset_index(drop=True), y.astype(int).to_numpy(), subject.astype(str).to_numpy(),
               {"source": str(f), "label": "progression by PLR2 (baseline features only)",
                "dropped": ["PLR3, MD-slope labels", "Total Visits (follow-up quantity)", "laterality",
                            "image references"], "note": "VF value -1 (not tested) -> missing"})


def load_fives_vascx(root=None):
    """FIVES fundus images with VascX vessel biomarkers: fives_vascx_dataset.csv, one row
    per image.  Label (plan): any disease (AMD, DR, glaucoma) vs normal.  The FIVES split
    column is ignored (we make our own).  No patient ids in the file: each image is
    treated as its own patient (caveat recorded)."""
    d = _root(root) / "fives_vascx"
    f = _need(d / "fives_vascx_dataset.csv", "copy fives_vascx_dataset.csv from the other project")
    D = pd.read_csv(f)
    codes = set(D.disease_code.astype(str).str.strip())
    _check(codes <= {"A", "D", "G", "N"}, f"FIVES: unexpected disease codes {sorted(codes)}")
    y = (D.disease_code.astype(str).str.strip() != "N").astype(int).to_numpy()
    X = D.drop(columns=["vascx_id", "image_id", "disease_code", "disease", "split"])
    return Raw(X, y, None, {"source": str(f), "label": "disease (AMD/DR/glaucoma) vs normal",
                            "dropped": ["ids", "disease labels", "FIVES split"],
                            "counts": D.disease.value_counts().to_dict(),
                            "caveat": "no patient ids: images treated as independent"})


# ---------------------------------------------------------------------------
# the Run 2 pool (55 datasets; plan Amendment 3)
# ---------------------------------------------------------------------------
# Same names and sources as Run 2 (archive: quantum_features/ladder_v1/FROZEN_LADDER_PROTOCOL.json).
# PMLB sets are read from data/pmlb_cache/<name>/<name>.tsv.gz (copied from Run 2's cache, so
# the files are the ones Run 2 used); iris and wine come with scikit-learn; heart-statlog,
# glass, yeast and tic-tac-toe are fetched from OpenML with the cache in data/openml_cache.

POOL = ["pmlb_analcatdata_boxing2", "pmlb_analcatdata_boxing1", "pmlb_corral", "pmlb_analcatdata_creditscore",
        "pmlb_irish", "pmlb_analcatdata_lawsuit", "pmlb_monk1", "pmlb_mux6", "pmlb_haberman", "pmlb_hayes_roth",
        "pmlb_monk3", "pmlb_appendicitis", "iris", "pmlb_monk2", "pmlb_prnn_synth", "pmlb_mofn_3_7_10",
        "pmlb_cloud", "pmlb_biomed", "pmlb_new_thyroid", "pmlb_breast_cancer", "pmlb_parity5+5", "pmlb_threeOf9",
        "pmlb_analcatdata_germangss", "pmlb_tae", "glass_binary", "pmlb_balance_scale", "pmlb_xd6",
        "pmlb_prnn_crabs", "pmlb_penguins", "pmlb_bupa", "heart", "pmlb_banana", "pmlb_glass2",
        "pmlb_breast_cancer_wisconsin_original", "pmlb_heart_disease_cleveland", "tic_tac_toe",
        "pmlb_car_evaluation", "pmlb_analcatdata_dmft", "pmlb_ecoli", "pmlb_saheart", "pmlb_cars", "pmlb_profb",
        "pmlb_nursery", "wine", "pmlb_led7", "pmlb_titanic", "pmlb_contraceptive_method", "yeast_binary",
        "pmlb_phoneme", "pmlb_mfeat_morphological", "pmlb_yeast", "pmlb_magic", "pmlb_wine_quality_red",
        "pmlb_sleep", "pmlb_page_blocks"]
# labels are known logical functions of the inputs (artificial rule problems): reported separately
POOL_BOOLEAN = ["pmlb_corral", "pmlb_monk1", "pmlb_monk2", "pmlb_monk3", "pmlb_mux6", "pmlb_parity5+5",
                "pmlb_threeOf9", "pmlb_xd6", "pmlb_mofn_3_7_10"]
POOL_MAX_N = 3000                      # Run 2's rule: larger datasets subsampled to 3000 rows


def binarize(target, name):
    """Two classes: positive = the later label in sorted order (1 over 0, 'present' over
    'absent').  More than two: positive = the LARGEST class, the rest negative (ties: the
    first in sorted order).  Every row is kept."""
    t = pd.Series(np.asarray(target)).astype(str)
    classes = sorted(t.unique())
    if len(classes) == 2:
        pos = classes[-1]
    else:
        counts = t.value_counts()
        pos = sorted(counts.index[counts == counts.max()])[0]
    return (t == pos).astype(int).to_numpy(), {"classes": len(classes), "positive_class": pos}


def _subsample(X, target, name, max_n=POOL_MAX_N):
    if len(X) <= max_n:
        return X, target, None
    from sklearn.model_selection import train_test_split
    import hashlib
    seed = int(hashlib.sha256(f"subsample_{name}".encode()).hexdigest()[:8], 16)
    idx = np.arange(len(X))
    try:
        keep, _ = train_test_split(idx, train_size=max_n, stratify=np.asarray(target), random_state=seed)
    except ValueError:                                      # a class too rare to stratify
        keep, _ = train_test_split(idx, train_size=max_n, random_state=seed)
    keep = np.sort(keep)
    return X.iloc[keep].reset_index(drop=True), np.asarray(target)[keep], len(target)


def _openml(name, version, root):
    try:
        from sklearn.datasets import fetch_openml
        b = fetch_openml(name=name, version=version, as_frame=True, parser="auto",
                         data_home=str(_root(root).parent / "openml_cache"))       # never the home folder
    except Exception as e:                                    # no network, or OpenML down
        raise NotReady(f"OpenML {name} v{version} could not be fetched: {e}")
    return b.frame, b.target_names[0] if getattr(b, "target_names", None) else b.frame.columns[-1]


def _legacy(name, root):
    from sklearn.datasets import load_iris, load_wine
    if name in ("iris", "wine"):
        d = (load_iris if name == "iris" else load_wine)()
        return pd.DataFrame(d.data, columns=d.feature_names), d.target, "scikit-learn"
    spec = {"heart": ("heart-statlog", 1, None), "glass_binary": ("glass", 1, "top2"),
            "yeast_binary": ("yeast", 1, "top2"), "tic_tac_toe": ("tic-tac-toe", 1, "codes")}[name]
    F, ycol = _openml(spec[0], spec[1], root)
    y = F[ycol].astype(str).to_numpy()
    X = F.drop(columns=[ycol])
    if spec[2] == "top2":                                   # Run 2: the two largest classes only
        vc = pd.Series(y).value_counts()
        keep = np.isin(y, vc.index[:2])
        X, y = X[keep].reset_index(drop=True), y[keep]
    if spec[2] == "codes":                                  # Run 2: board cells as category codes
        X = X.apply(lambda c: c.astype("category").cat.codes.astype(float))
    return X, y, f"OpenML {spec[0]} v{spec[1]}"


def load_pool(name, root=None):
    root = _root(root)
    if name.startswith("pmlb_"):
        pm = name[len("pmlb_"):]
        f = _need(root.parent / "pmlb_cache" / pm / f"{pm}.tsv.gz",
                  "copy Run 2's pmlb_cache into data/pmlb_cache (see the README / plan Amendment 3)")
        D = pd.read_csv(f, sep="\t")
        target = D.pop("target").to_numpy()
        X, src = D, f"PMLB {pm} (Run 2 cache)"
    else:
        X, target, src = _legacy(name, root)
    X, target, n_raw = _subsample(pd.DataFrame(X).reset_index(drop=True), target, name)
    y, binfo = binarize(target, name)
    return Raw(X, y, None, {"source": src, "label": "binary as is" if binfo["classes"] == 2
                            else f"largest class ({binfo['positive_class']}) vs rest", **binfo,
                            "subsampled_from": n_raw, "boolean_rule_dataset": name in POOL_BOOLEAN})


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
    "papila": load_papila,
    "grape": load_grape,
    "keratoconus": _todo("keratoconus", "needs the file's columns and class coding"),
    "perg_ioba": _todo("perg_ioba", "needs P50/N95 features from the signals, first visit per subject"),
    "fives_vascx": load_fives_vascx,
}


def load(name, root=None) -> Raw:
    root = _root(root)
    if name in ADAPTERS:
        return ADAPTERS[name](root=root)
    if name in POOL:
        return load_pool(name, root)
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
