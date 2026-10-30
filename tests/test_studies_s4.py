"""S4 (VQC), NumPy reference and scoring: the checks of PLAN.md Section 6 that need no PyTorch."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, evolve_batch_strang, strang_steps  # noqa: E402
from studies.common import vqc_np as V  # noqa: E402

_spec = importlib.util.spec_from_file_location("s4run", ROOT / "studies" / "s4_vqc" / "run.py")
S4 = importlib.util.module_from_spec(_spec)
sys.modules["s4run"] = S4
_spec.loader.exec_module(S4)


@pytest.mark.parametrize("p,J,tau", [(2, 1.0, 1.0), (3, 0.7, 2.0), (4, 4.0 / 3, 5.0)])
def test_informed_circuit_with_the_anneal_schedule_is_the_exact_map(p, J, tau):
    X = np.random.default_rng(p).standard_normal((5, p))
    n = strang_steps(tau)
    g, e, b = V.strang_schedule(J, tau, n)
    psi = V.informed_forward(X, g, e, b)
    assert np.abs(psi - evolve_batch_strang(X, J, tau, n)).max() < 1e-10
    assert np.abs(V.readout(psi) - compute_bank(X, J, tau, method="strang")[:, :3 * p]).max() < 1e-6


def test_per_qubit_informed_equals_shared_when_values_agree():
    p, L = 3, 4
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, p))
    g, e, b = rng.uniform(0, 1, L), rng.uniform(0, 1, L), rng.uniform(0, 1, L)
    a = V.informed_forward(X, g, e, b)
    c = V.informed_forward(X, np.repeat(g[:, None], p, 1), e, np.repeat(b[:, None], p, 1))
    assert np.allclose(a, c)


def _cnot(psi, c, t):
    idx = np.arange(psi.shape[1])
    return psi[:, idx ^ (((idx >> c) & 1) << t)]


@pytest.mark.parametrize("p", [2, 3, 5])
def test_cnot_ring_permutation_equals_gate_by_gate(p):
    psi = np.random.default_rng(p).standard_normal((3, 1 << p)) + 0j
    ref = psi
    pairs = [(i, (i + 1) % p) for i in range(p)] if p > 2 else [(0, 1), (1, 0)]
    for c, t in pairs:
        ref = _cnot(ref, c, t)
    assert np.allclose(psi[:, V.cnot_ring_perm(p)], ref)


def test_generic_circuit_is_unitary_and_encodes():
    p = 4
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, p))
    th = rng.uniform(0, 2 * np.pi, (3, p, 2))
    for re in (False, True):
        psi = V.generic_forward(X, th, re)
        assert np.allclose((np.abs(psi) ** 2).sum(1), 1)
    # zero angles, one layer: RY(x) on |0>, then the CNOT ring permutes basis states
    psi = V.generic_forward(X[:1], np.zeros((1, p, 2)), False)
    single = np.ones(1, complex)
    for q in reversed(range(p)):                   # qubit q is bit q: kron(high ... low)
        single = np.kron(single, [np.cos(X[0, q] / 2), np.sin(X[0, q] / 2)])
    assert np.allclose(psi[0], single[V.cnot_ring_perm(p)])


def test_jobs_cover_the_plan():
    J = S4.jobs(S4.DESIGN, False)
    assert len(J) == 4 * 5 * 3 + 4 * 5
    assert {j[0] for j in J} == {"G1", "G2", "H1", "H2"} and {j[1] for j in J} == {1, 2, 4, 8, 16}
    assert {(j[2], j[3]) for j in J} == {(4, 3000), (6, 3000), (8, 3000), (8, 30000)}


def test_score_end_to_end_on_fake_files(tmp_path, monkeypatch, capsys):
    out, s1 = tmp_path / "s4", tmp_path / "s1"
    (out / "train").mkdir(parents=True)
    s1.mkdir()
    monkeypatch.setattr(S4, "OUT", out)
    monkeypatch.setattr(S4.S1, "OUT", s1)
    M = pd.DataFrame([t for p in (4, 6, 8) for t in S4.S1.teacher_specs(p)])
    M.to_csv(s1 / "S1_TEACHERS.csv", index=False)
    B = []
    for nm in M.name:
        for n in (3000, 30000):
            B.append(dict(choice="learned", name=nm, n=n, ref_auc=0.95, test_auc=0.85))
    pd.DataFrame(B).to_csv(s1 / "S1_PARTB.csv", index=False)
    rows = []
    for t in M.itertuples():
        for kind in ("G1", "G2", "H1", "H2"):
            for L in (1, 2, 4, 8, 16):
                for n in ((3000, 30000) if t.p == 8 else (3000,)):
                    auc = 0.85 if kind[0] == "G" else 0.80 + 0.01 * L          # H improves with L
                    rows.append(dict(name=t.name, kind=kind, L=L, p=t.p, n=n, params=10 * L,
                                     val_loss=1 - auc, val_auc=auc, test_auc=auc, best_step=100, steps=500))
    pd.DataFrame(rows).to_csv(out / "train" / "fake.csv", index=False)
    S4.do_score()
    K = pd.read_csv(out / "S4_CHECKS.csv").set_index("check")
    # H at L = 16: 0.96 >= open-book 0.95 -> parity everywhere; G = classical exactly
    assert bool(K.loc["V0", "held"]) and bool(K.loc["V1", "held"]) and bool(K.loc["V2", "held"]) and bool(K.loc["V3", "held"])
    T = pd.read_csv(out / "S4_SELECTED.csv")
    assert (T[T.kind == "H1"].L == 16).all()                    # L picked on validation
    assert "S4 checks" in capsys.readouterr().out
