"""
GPU learners for S1 Part B (PyTorch).  Every function trains MANY independent models at once
(one per teacher), because all teachers at one p share the same inputs: only the labels differ.

  logreg_path   L2 logistic regression on a fixed feature matrix, for a grid of penalties,
                every teacher at once (L-BFGS on the sum of the independent objectives).
                Used for the fixed bases (polynomial, Fourier, random Fourier) and for the
                open-book reference (the teacher's own <X>,<Y>,<Z> features).
  train_mlps    a batch of independent fully connected ReLU networks (one per teacher), Adam,
                early stopping on each network's own validation loss.

Both return, per model, the validation loss, validation AUC and test AUC of the setting each
model picks on its validation rows (the penalty for logreg_path, the checkpoint for
train_mlps).  Nothing here looks at the test rows to make a choice.

tests/test_studies_s1b.py checks both against plain per-model versions (and logreg against
scikit-learn) before anything relies on them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def _aucs(Y, S):
    """AUC per column; nan where a column has one class"""
    out = np.full(Y.shape[1], np.nan)
    for m in range(Y.shape[1]):
        if 0 < Y[:, m].sum() < len(Y):
            out[m] = roc_auc_score(Y[:, m], S[:, m])
    return out


def _bce_cols(logits, Y):
    """mean binary cross-entropy per column: (n, M), (n, M) -> (M,)"""
    return F.binary_cross_entropy_with_logits(logits, Y, reduction="none").mean(0)


# ---------------------------------------------------------------------------
# L2 logistic regression, many label sets at once
# ---------------------------------------------------------------------------

def _standardizer(Xtr):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True)
    sd = torch.where(sd > 1e-9, sd, torch.ones_like(sd))      # constant columns become 0
    return mu, sd


def logreg_path(Xtr, Ytr, Xva, Yva, Xte, Yte, lams, device, max_iter=300, X_per_model=False):
    """
    Xtr (n, D) shared by all models, or (M, n, D) one per model when X_per_model.
    Y* (n, M) 0/1.  For each penalty lam (largest first, warm-started) minimise, per model m,
        mean_i BCE(x_i . w_m + b_m, y_im) + lam/2 * |w_m|^2      (intercept not penalised)
    on standardized features (train-row mean and sd).
    Returns dict of (M,) arrays: lam, val_loss, val_auc, test_auc, at each model's best lam
    (lowest validation loss), plus 'path' = list of per-lam dicts.
    """
    f32 = torch.float32
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=f32, device=device)
    Xtr, Xva, Xte = T(Xtr), T(Xva), T(Xte)
    Ytr_t, Yva_t = T(Ytr), T(Yva)
    if X_per_model:
        mu = Xtr.mean(1, keepdim=True)
        sd = Xtr.std(1, keepdim=True)
        sd = torch.where(sd > 1e-9, sd, torch.ones_like(sd))
        M, _, D = Xtr.shape
        lin = lambda X, W, b: torch.einsum("mnd,dm->nm", X, W) + b
    else:
        mu, sd = _standardizer(Xtr)
        D, M = Xtr.shape[1], Ytr.shape[1]
        lin = lambda X, W, b: X @ W + b
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(D, M, dtype=f32, device=device, requires_grad=True)
    b = torch.zeros(M, dtype=f32, device=device, requires_grad=True)
    path = []
    for lam in sorted(lams, reverse=True):
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = _bce_cols(lin(Xtr, W, b), Ytr_t).sum() + 0.5 * lam * (W * W).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            sva = lin(Xva, W, b)
            ste = lin(Xte, W, b)
            path.append(dict(lam=lam, val_loss=_bce_cols(sva, Yva_t).cpu().numpy(),
                             val_auc=_aucs(Yva, sva.cpu().numpy()),
                             test_auc=_aucs(Yte, ste.cpu().numpy())))
    V = np.stack([r["val_loss"] for r in path])                 # (n_lams, M)
    V = np.where(np.isfinite(V), V, np.inf)
    k = V.argmin(0)
    pick = lambda key: np.array([path[k[m]][key][m] for m in range(len(k))])
    return dict(lam=np.array([path[i]["lam"] for i in k]), val_loss=pick("val_loss"),
                val_auc=pick("val_auc"), test_auc=pick("test_auc"), path=path)


# ---------------------------------------------------------------------------
# independent MLPs, trained as one batch
# ---------------------------------------------------------------------------

def mlp_params(p, width, depth):
    """number of trainable parameters of one network: p -> width (x depth hidden layers) -> 1"""
    return (p * width + width) + (depth - 1) * (width * width + width) + (width + 1)


class BatchedMLP:
    """M independent ReLU networks p -> width -> ... -> width -> 1 (depth hidden layers).
    Weights (M, fan_in, fan_out); initialised like torch.nn.Linear (uniform +-1/sqrt(fan_in))."""

    def __init__(self, M, p, width, depth, device, generator):
        dims = [p] + [width] * depth + [1]
        self.params = []
        for i, o in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(i)
            W = (torch.rand(M, i, o, generator=generator, device=device) * 2 - 1) * bound
            b = (torch.rand(M, 1, o, generator=generator, device=device) * 2 - 1) * bound
            self.params += [W.requires_grad_(), b.requires_grad_()]

    def logits(self, X, params=None):
        """X (B, p) shared by all networks -> (B, M)"""
        P = self.params if params is None else params
        h = torch.einsum("bi,mio->mbo", X, P[0]) + P[1]
        for k in range(2, len(P), 2):
            h = torch.relu(h)
            h = torch.baddbmm(P[k + 1], h, P[k])
        return h[..., 0].T


def train_mlps(Xtr, Ytr, Xva, Yva, Xte, Yte, width, depth, device, seed,
               lr=1e-3, batch=512, eval_every=50, patience=40, max_steps=20_000, chunk=2048):
    """
    One network per column of Y, all trained together (same minibatch rows, independent
    weights).  Adam on the summed per-network mean BCE.  Every eval_every steps, each network's
    validation loss is computed; each keeps its own best checkpoint, and training stops once
    every network has gone `patience` evaluations without improving (or at max_steps).
    Returns dict of (M,) arrays at each network's best checkpoint: val_loss, val_auc, test_auc,
    best_step; plus steps (total steps run) and params (per network).
    """
    f32 = torch.float32
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=f32, device=device)
    Xtr_t, Ytr_t, Xva_t, Yva_t, Xte_t = T(Xtr), T(Ytr), T(Xva), T(Yva), T(Xte)
    n, p = Xtr_t.shape
    M = Ytr_t.shape[1]
    g = torch.Generator(device=device).manual_seed(int(seed))
    net = BatchedMLP(M, p, width, depth, device, g)
    opt = torch.optim.Adam(net.params, lr=lr)
    best = [q.detach().clone() for q in net.params]
    best_val = torch.full((M,), float("inf"), device=device)
    best_step = torch.zeros(M, dtype=torch.long, device=device)
    stale = torch.zeros(M, dtype=torch.long, device=device)
    bs = min(batch, n)

    def chunked(X, params=None):
        with torch.no_grad():
            return torch.cat([net.logits(X[i:i + chunk], params) for i in range(0, len(X), chunk)])

    step = 0
    perm, pos = torch.randperm(n, generator=g, device=device), 0
    while step < max_steps:
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g, device=device), 0
        idx = perm[pos:pos + bs]
        pos += bs
        opt.zero_grad(set_to_none=True)
        loss = _bce_cols(net.logits(Xtr_t[idx]), Ytr_t[idx]).sum()
        loss.backward()
        opt.step()
        step += 1
        if step % eval_every == 0 or step == max_steps:
            vl = _bce_cols(chunked(Xva_t), Yva_t)
            vl = torch.where(torch.isfinite(vl), vl, torch.full_like(vl, float("inf")))
            better = vl < best_val - 1e-6
            with torch.no_grad():
                for q, bq in zip(net.params, best):
                    m = better.view(-1, *([1] * (q.dim() - 1)))
                    bq.copy_(torch.where(m, q, bq))
            best_val = torch.where(better, vl, best_val)
            best_step = torch.where(better, torch.full_like(best_step, step), best_step)
            stale = torch.where(better, torch.zeros_like(stale), stale + 1)
            if bool((stale >= patience).all()):
                break
    sva = chunked(Xva_t, best).cpu().numpy()
    ste = chunked(Xte_t, best).cpu().numpy()
    return dict(val_loss=best_val.cpu().numpy(), val_auc=_aucs(Yva, sva), test_auc=_aucs(Yte, ste),
                best_step=best_step.cpu().numpy(), steps=step, params=mlp_params(p, width, depth))
