"""The PyTorch circuits (studies/common/torchvqc.py) against the NumPy reference (vqc_np.py).

Skipped where PyTorch isn't installed; run on the GPU machine before S4 trains there.  The GPU
cases are skipped when no GPU is visible."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, strang_steps  # noqa: E402
from studies.common import torchvqc as tv  # noqa: E402
from studies.common import vqc_np as V  # noqa: E402

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _circ(kind, L, M, p, dev):
    return tv.Circuits(kind, L, M, p, dev, torch.Generator(device=dev).manual_seed(0))


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kind", ["G1", "G2", "H1", "H2"])
def test_torch_forward_matches_numpy(dev, kind):
    p, L, M = 3, 3, 2
    X = np.random.default_rng(1).standard_normal((5, p))
    c = _circ(kind, L, M, p, dev)
    with torch.no_grad():
        if kind[0] == "H":                       # move off the ramp start so every parameter matters
            for q in c.params:
                q.add_(torch.rand(q.shape, generator=torch.Generator(device=dev).manual_seed(3), device=dev))
        F_t = c.readout(c.states(torch.as_tensor(X, dtype=torch.float32, device=dev))).cpu().numpy()
    P = [q.detach().cpu().numpy() for q in c.params]
    for m in range(M):
        if kind[0] == "G":
            psi = V.generic_forward(X, P[0][m], kind == "G2")
        else:
            psi = V.informed_forward(X, P[0][m], P[1][m], P[2][m])
        assert np.abs(F_t[m] - V.readout(psi)).max() < 1e-5


@pytest.mark.parametrize("dev", DEVICES)
def test_torch_informed_with_anneal_schedule_is_the_exact_map(dev):
    p, J, tau = 3, 0.7, 2.0
    n = strang_steps(tau)
    g, e, b = V.strang_schedule(J, tau, n)
    c = _circ("H1", len(e), 1, p, dev)
    T = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32, device=dev)
    X = np.random.default_rng(2).standard_normal((4, p))
    with torch.no_grad():
        F_t = c.readout(c.states(torch.as_tensor(X, dtype=torch.float32, device=dev), [T(g), T(e), T(b)])).cpu().numpy()[0]
    assert np.abs(F_t - compute_bank(X, J, tau, method="strang")[:, :3 * p]).max() < 2e-4   # float32 over 801 layers


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kind", ["G2", "H1"])
def test_training_learns_an_easy_label(dev, kind):
    rng = np.random.default_rng(0)
    p = 3
    X = rng.standard_normal((1500, p)).astype(np.float32)
    Y = np.stack([(X[:, 0] > 0), (X[:, 1] + X[:, 2] > 0)], 1).astype(np.float32)
    r = tv.train_circuits(kind, 2, X[:1000], Y[:1000], X[1000:1250], Y[1000:1250], X[1250:], Y[1250:], dev,
                          seed=0, eval_every=20, patience=10, max_steps=600)
    assert (r["test_auc"] > 0.8).all() and r["steps"] <= 600 and np.isfinite(r["val_loss"]).all()
