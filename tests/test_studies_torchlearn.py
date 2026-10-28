"""The GPU learners for S1 Part B (studies/common/torchlearn.py) against plain versions.

Skipped where PyTorch isn't installed (e.g. a laptop); run on the GPU machine before Part B
runs there.  The GPU cases are skipped when no GPU is visible."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.common import torchlearn as tl  # noqa: E402

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _data(n, p, M, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    W = rng.standard_normal((p, M))
    Y = (rng.random((n, M)) < 1 / (1 + np.exp(-(X @ W)))).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("dev", DEVICES)
def test_logreg_path_matches_sklearn(dev):
    from sklearn.linear_model import LogisticRegression
    X, Y = _data(400, 5, 3)
    Xs = (X[:300] - X[:300].mean(0)) / X[:300].std(0, ddof=1)
    for lam in (1e-1, 1e-3):
        res = tl.logreg_path(X[:300], Y[:300], X[300:350], Y[300:350], X[350:], Y[350:], [lam], dev)
        for m in range(3):
            sk = LogisticRegression(C=1.0 / (lam * 300), tol=1e-10, max_iter=10_000).fit(Xs, Y[:300, m])
            Xte = (X[350:] - X[:300].mean(0)) / X[:300].std(0, ddof=1)
            from sklearn.metrics import roc_auc_score
            assert abs(roc_auc_score(Y[350:, m], sk.decision_function(Xte)) - res["test_auc"][m]) < 1e-3


@pytest.mark.parametrize("dev", DEVICES)
def test_logreg_per_model_features_equals_shared(dev):
    X, Y = _data(300, 4, 2, seed=1)
    a = tl.logreg_path(X[:200], Y[:200], X[200:250], Y[200:250], X[250:], Y[250:], [1e-2], dev)
    Xm = np.stack([X, X])
    b = tl.logreg_path(Xm[:, :200], Y[:200], Xm[:, 200:250], Y[200:250], Xm[:, 250:], Y[250:], [1e-2], dev,
                       X_per_model=True)
    assert np.allclose(a["test_auc"], b["test_auc"], atol=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_batched_mlp_equals_separate_networks(dev):
    g = torch.Generator(device=dev).manual_seed(0)
    net = tl.BatchedMLP(3, 5, 7, 3, dev, g)
    X = torch.randn(11, 5, device=dev)
    out = net.logits(X)
    assert out.shape == (11, 3)
    for m in range(3):
        h = X
        P = [q[m] for q in net.params]
        for k in range(0, len(P), 2):
            h = h @ P[k] + P[k + 1][0]
            if k + 2 < len(P):
                h = torch.relu(h)
        assert torch.allclose(out[:, m], h[:, 0], atol=1e-5)
    assert sum(q[0].numel() for q in net.params) == tl.mlp_params(5, 7, 3)


@pytest.mark.parametrize("dev", DEVICES)
def test_train_mlps_learns_and_keeps_best_checkpoint(dev):
    X, Y = _data(3_000, 4, 2, seed=2)
    r = tl.train_mlps(X[:2000], Y[:2000], X[2000:2500], Y[2000:2500], X[2500:], Y[2500:], 16, 2, dev, seed=0,
                      eval_every=20, patience=10, max_steps=2_000)
    assert (r["test_auc"] > 0.75).all() and (r["best_step"] > 0).all() and r["steps"] <= 2_000
    assert np.isfinite(r["val_loss"]).all()
