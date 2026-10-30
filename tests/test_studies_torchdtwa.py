"""The PyTorch DTWA (studies/common/torchdtwa.py) against the NumPy one (dtwa.py, itself tested against
three exact limits).  Skipped where PyTorch isn't installed; GPU cases skipped without a GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import strang_steps  # noqa: E402
from studies.common import dtwa  # noqa: E402
from studies.common import torchdtwa as td  # noqa: E402

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("p,J,tau", [(3, 0.7, 3.0), (5, 1.0, 10.0)])
def test_torch_evolve_matches_numpy(dev, p, J, tau):
    X = np.random.default_rng(p).standard_normal((4, p))
    S0 = dtwa.initial_points(p, 64, np.random.default_rng(1))
    n = strang_steps(tau)
    ref = dtwa.evolve(X, S0, J, tau, n)
    S64 = td.evolve_torch(X, S0, J, tau, n, dev).cpu().numpy()     # the default is float64
    assert td.evolve_torch(X[:1], S0[:2], J, tau, 2, dev).dtype == torch.float64
    assert np.abs(S64 - ref).max() < 1e-9
    # (float32 is NOT checked per trajectory: the trajectories are chaotic, and rounding grows to ~0.03)


@pytest.mark.parametrize("dev", DEVICES)
def test_bank_prefix_and_nesting(dev):
    p, J, tau = 4, 4.0 / 3, 5.0
    X = np.random.default_rng(0).standard_normal((6, p))
    n = strang_steps(tau)
    R = td.nested_banks(X, J, tau, n, [16, 64, 256], np.random.default_rng(2), dev, chunk=4, dtype=torch.float64)
    assert R["M_used"] == 256 and sorted(k for k in R if isinstance(k, int)) == [16, 64, 256]
    full, _, _ = dtwa.dtwa_bank(X, J, tau, n, M=256)       # all 4^4 points: order doesn't matter
    assert np.abs(R[256] - full).max() < 1e-6
    S0 = td.starting_points(p, 256, np.random.default_rng(2))
    sub = dtwa.bank_from_trajectories(dtwa.evolve(X, S0[:16], J, tau, n))
    assert np.abs(R[16] - sub).max() < 1e-9               # prefix = the first 16 shuffled points
    assert R["loc_var"].shape == (6, 3 * p) and (R["loc_var"] >= 0).all()


def test_starting_points():
    E = td.starting_points(3, 10_000, np.random.default_rng(0))
    assert E.shape == (64, 3, 3) and len({tuple(e.ravel()) for e in E}) == 64
    R = td.starting_points(9, 1000, np.random.default_rng(0))
    assert R.shape == (1000, 9, 3) and (R[..., 0] == 1).all()
