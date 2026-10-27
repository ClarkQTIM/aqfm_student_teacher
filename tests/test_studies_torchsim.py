"""The PyTorch simulator (studies/common/torchsim.py) must agree with the NumPy one.

Skipped where PyTorch isn't installed (e.g. a laptop); run on the GPU machine before any S1
bank is simulated there.  The GPU test is skipped when no GPU is visible."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqfm_prescreen.aqfm import compute_bank, evolve_batch_strang, observables_batch, strang_steps  # noqa: E402
from studies.common import bankstore as bs  # noqa: E402
from studies.common.torchsim import compute_bank_torch, evolve_strang_torch, observables_torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("s1run_t", ROOT / "studies" / "s1_scaling" / "run.py")
S1 = importlib.util.module_from_spec(_spec)
sys.modules["s1run_t"] = S1
_spec.loader.exec_module(S1)


def _cases():
    rng = np.random.default_rng(0)
    Jm = rng.standard_normal((5, 5)) * 0.4
    return [(3, 0.7, 2.0), (4, 4.0 / 3, 5.0), (5, Jm, 3.0), (6, 0.4, 20.0)]


@pytest.mark.parametrize("p,J,tau", _cases())
def test_states_and_observables_match_numpy_in_float64(p, J, tau):
    Z = bs.pool_inputs(p)[:25]
    n = strang_steps(tau)
    want = observables_batch(evolve_batch_strang(Z, J, tau, n))
    got = observables_torch(evolve_strang_torch(Z, J, tau, n, "cpu")).numpy()
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-10


def test_bank_matches_compute_bank():
    Z = bs.pool_inputs(4)[:30]
    a = compute_bank(Z, 2.0 / 3, 5.0, method="strang")
    b = compute_bank_torch(Z, 2.0 / 3, 5.0, device="cpu", chunk=7)        # several chunks
    assert a.dtype == b.dtype == np.float32 and np.array_equal(a.shape, b.shape)
    assert np.abs(a.astype(float) - b.astype(float)).max() < 1e-6        # float32 rounding only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible")
def test_gpu_matches_numpy():
    p, J, tau = 8, 4.0 / 7, 20.0
    Z = bs.pool_inputs(p)[:40]
    n = strang_steps(tau)
    want = observables_batch(evolve_batch_strang(Z, J, tau, n))
    got = observables_torch(evolve_strang_torch(Z, J, tau, n, "cuda:0")).cpu().numpy()
    assert np.abs(got - want).max() < 1e-9


def test_gpu_job_writes_the_same_blocks_as_the_cpu_path(tmp_path, monkeypatch):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    monkeypatch.setattr(bs, "BLOCK", 10)
    monkeypatch.setenv(bs.TORCH_ENV, dev)
    st, _ = S1.gpu_job((tmp_path, 3, 2.0, 3.0, [0, 2]))
    assert st.startswith("ok")
    for b in (0, 2):
        got = np.load(bs.block_path(tmp_path, 3, "Q", 2.0, 3.0, b))
        want = compute_bank(bs.pool_inputs(3)[b * 10:(b + 1) * 10], 1.0, 3.0, method="strang")
        assert np.abs(got.astype(float) - want.astype(float)).max() < 1e-6
    assert not bs.block_path(tmp_path, 3, "Q", 2.0, 3.0, 1).exists()
    assert S1.gpu_job((tmp_path, 3, 2.0, 3.0, [0, 2]))[0] == "skip"
