"""
PyTorch (GPU) circuits for S4, trained end to end: one circuit + linear readout per teacher, all
teachers at once (they share the inputs; only the labels differ).

Gate for gate the same as studies/common/vqc_np.py (the NumPy reference, itself tested against
the exact anneal); tests/test_studies_torchvqc.py checks the forward passes agree to 1e-5.

States are (M, B, 2^p) complex64: M models (teachers), B rows.  A single-qubit gate is given by
its four entries, each broadcastable to (M, B).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from aqfm_prescreen.aqfm import z_signs
from studies.common.vqc_np import cnot_ring_perm

CDT = torch.complex64


def _aucs(Y, S):
    out = np.full(Y.shape[1], np.nan)
    for m in range(Y.shape[1]):
        if 0 < Y[:, m].sum() < len(Y):
            out[m] = roc_auc_score(Y[:, m], S[:, m])
    return out


def apply_1q(psi, u00, u01, u10, u11, q, p):
    """psi (M, B, 2^p); u.. broadcastable to (M, B)"""
    M, B, _ = psi.shape
    a = psi.reshape(M, B, 1 << (p - 1 - q), 2, 1 << q)
    a0, a1 = a[:, :, :, 0, :], a[:, :, :, 1, :]
    e = lambda u: u.reshape(u.shape + (1, 1)) if torch.is_tensor(u) and u.dim() == 2 else u
    u00, u01, u10, u11 = e(u00), e(u01), e(u10), e(u11)
    return torch.stack([u00 * a0 + u01 * a1, u10 * a0 + u11 * a1], dim=3).reshape(M, B, -1)


def _c(x):
    return x.to(CDT)


class Circuits:
    """the four S4 students.  kind in {G1, G2, H1, H2}; L layers; M models; p qubits."""

    def __init__(self, kind, L, M, p, device, generator):
        self.kind, self.L, self.M, self.p, self.dev = kind, L, M, p, device
        f32 = torch.float32
        if kind in ("G1", "G2"):
            th = torch.rand(M, L, p, 2, generator=generator, device=device) * (2 * math.pi)
            self.params = [th.requires_grad_()]
            self.perm = torch.as_tensor(cnot_ring_perm(p), device=device)
        else:                                     # linear-ramp start [Zhou et al. 2020], T0 = 5, c = 1
            s = (torch.arange(L, dtype=f32, device=device) + 0.5) / L
            T0 = 5.0
            g = (s * T0 / L).repeat(M, 1)                                  # (M, L)
            b = ((1 - s) * T0 / L).repeat(M, 1)
            eta = g / max(p - 1, 1)
            if kind == "H2":
                g = g[..., None].repeat(1, 1, p)
                b = b[..., None].repeat(1, 1, p)
            self.params = [g.clone().requires_grad_(), eta.clone().requires_grad_(), b.clone().requires_grad_()]
            ZS = torch.as_tensor(z_signs(p), dtype=f32, device=device)    # (2^p, p)
            self.ZS = ZS
            self.zz = 0.5 * (ZS.sum(1) ** 2 - p)                           # (2^p,)
        ZS = torch.as_tensor(z_signs(p), dtype=f32, device=device)
        self.ZSr = ZS
        idx = torch.arange(1 << p, device=device)
        self.flip = [idx ^ (1 << q) for q in range(p)]
        self.sgn = [(1.0 - 2.0 * ((idx >> q) & 1)).to(f32) for q in range(p)]

    def n_params(self):
        return int(sum(q[0].numel() for q in self.params))

    def states(self, X, params=None):
        """X (B, p) float tensor -> (M, B, 2^p)"""
        P = self.params if params is None else params
        M, p = self.M, self.p
        B = X.shape[0]
        if self.kind in ("G1", "G2"):
            th = P[0]
            psi = torch.zeros(M, B, 1 << p, dtype=CDT, device=self.dev)
            psi[:, :, 0] = 1.0
            cx, sx = torch.cos(X / 2), torch.sin(X / 2)                    # (B, p) encoding RY(x)
            for layer in range(self.L):
                if layer == 0 or self.kind == "G2":
                    for q in range(p):
                        c, s = _c(cx[:, q])[None], _c(sx[:, q])[None]       # (1, B)
                        psi = apply_1q(psi, c, -s, s, c, q, p)
                for q in range(p):
                    a, z = th[:, layer, q, 0], th[:, layer, q, 1]           # (M,)
                    c, s = torch.cos(a / 2), torch.sin(a / 2)
                    em, ep = torch.polar(torch.ones_like(z), -z / 2), torch.polar(torch.ones_like(z), z / 2)
                    # RZ(z) RY(a) = [[e^{-iz/2} c, -e^{-iz/2} s], [e^{iz/2} s, e^{iz/2} c]]
                    u00, u01, u10, u11 = em * c, -em * s, ep * s, ep * c
                    psi = apply_1q(psi, u00[:, None], u01[:, None], u10[:, None], u11[:, None], q, p)
                psi = psi[:, :, self.perm]
            return psi
        g, eta, b = P
        psi = torch.full((M, B, 1 << p), 2.0 ** (-p / 2), dtype=CDT, device=self.dev)
        for layer in range(self.L):
            if self.kind == "H1":
                data = g[:, layer][:, None, None] * (X @ self.ZS.T)[None]          # (M, B, 2^p)
            else:
                data = (X[None] * g[:, layer][:, None, :]) @ self.ZS.T             # (M, B, 2^p)
            ang = data + eta[:, layer][:, None, None] * self.zz[None, None]
            psi = psi * torch.polar(torch.ones_like(ang), ang)
            for q in range(p):
                bq = b[:, layer] if self.kind == "H1" else b[:, layer, q]          # (M,)
                c, s = _c(torch.cos(bq))[:, None], 1j * _c(torch.sin(bq))[:, None]
                psi = apply_1q(psi, c, s, s, c, q, p)
        return psi

    def readout(self, psi):
        """(M, B, 3p): <X_q>, <Y_q>, <Z_q> in the bank conventions (as observables_batch)"""
        prob = psi.real ** 2 + psi.imag ** 2
        conj = psi.conj()
        Xs, Ys = [], []
        for q in range(self.p):
            pf = psi[:, :, self.flip[q]]
            Xs.append((conj * pf).real.sum(-1))
            Ys.append((conj * (1j * self.sgn[q]) * pf).real.sum(-1))
        Z = prob @ self.ZSr
        return torch.cat([torch.stack(Xs, -1), torch.stack(Ys, -1), Z], -1)


def train_circuits(kind, L, Xtr, Ytr, Xva, Yva, Xte, Yte, device, seed,
                   lr=0.01, batch=256, eval_every=25, patience=40, max_steps=10_000, chunk=1024):
    """one circuit + linear readout (3p weights + bias) per column of Y, all trained together
    (same minibatch rows); Adam on the summed per-model mean BCE; per-model early stopping on
    validation loss, best checkpoint kept.  Returns (M,) arrays at each model's best checkpoint:
    val_loss, val_auc, test_auc, best_step; plus steps and params (circuit + readout)."""
    f32 = torch.float32
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=f32, device=device)
    Xtr_t, Ytr_t, Xva_t, Yva_t, Xte_t = T(Xtr), T(Ytr), T(Xva), T(Yva), T(Xte)
    n, p = Xtr_t.shape
    M = Ytr_t.shape[1]
    g = torch.Generator(device=device).manual_seed(int(seed))
    circ = Circuits(kind, L, M, p, device, g)
    w = (torch.randn(M, 3 * p, generator=g, device=device) * 0.1).requires_grad_()
    b0 = torch.zeros(M, device=device, requires_grad=True)
    params = circ.params + [w, b0]
    opt = torch.optim.Adam(params, lr=lr)
    best = [q.detach().clone() for q in params]
    best_val = torch.full((M,), float("inf"), device=device)
    best_step = torch.zeros(M, dtype=torch.long, device=device)
    stale = torch.zeros(M, dtype=torch.long, device=device)
    bs = min(batch, n)

    def logits(X, P):
        Fx = circ.readout(circ.states(X, P[:-2]))                 # (M, B, 3p)
        return ((Fx * P[-2][:, None, :]).sum(-1) + P[-1][:, None]).T   # (B, M)

    def chunked(X, P):
        with torch.no_grad():
            return torch.cat([logits(X[i:i + chunk], P) for i in range(0, len(X), chunk)])

    def bce(lg, Y):
        return F.binary_cross_entropy_with_logits(lg, Y, reduction="none").mean(0)

    step = 0
    perm, pos = torch.randperm(n, generator=g, device=device), 0
    while step < max_steps:
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g, device=device), 0
        idx = perm[pos:pos + bs]
        pos += bs
        opt.zero_grad(set_to_none=True)
        loss = bce(logits(Xtr_t[idx], params), Ytr_t[idx]).sum()
        loss.backward()
        opt.step()
        step += 1
        if step % eval_every == 0 or step == max_steps:
            vl = bce(chunked(Xva_t, params), Yva_t)
            vl = torch.where(torch.isfinite(vl), vl, torch.full_like(vl, float("inf")))
            better = vl < best_val - 1e-6
            with torch.no_grad():
                for q, bq in zip(params, best):
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
                best_step=best_step.cpu().numpy(), steps=step, params=circ.n_params() + 3 * p + 1)
