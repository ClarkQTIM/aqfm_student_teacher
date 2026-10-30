"""
NumPy reference for the S4 circuits (studies/s4_vqc/PLAN.md, Section 2): forward pass only.

It is the ground truth for the PyTorch version (studies/common/torchvqc.py, which trains), and it
is checked against the exact anneal locally: with its layers set to the map's own Strang steps,
the informed circuit IS the exact map (tests/test_studies_s4.py).

Conventions are aqfm_prescreen.aqfm's: qubit q is bit q of the basis index, Z eigenvalue
1 - 2 bit; the readout is observables_batch (same X, Y, Z conventions as every bank).

Gates (single-qubit, acting on (|0>, |1>) amplitudes):
  RY(t) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
  RZ(t) = diag(e^{-it/2}, e^{+it/2})
  e^{i b X} = [[cos b, i sin b], [i sin b, cos b]]          (the driver for a time b)
  e^{i (sum_i g_i x_i Z_i + eta sum_{i<j} Z_i Z_j)}         (data + coupling for a time; diagonal)

Students
  generic:  start |0>; G1 encodes RY(x_i) once, G2 before every layer; layer l applies
            RZ(theta[l,i,1]) RY(theta[l,i,0]) on each qubit, then the CNOT ring 0->1->...->p-1->0.
  informed: start |+>; layer l applies the diagonal with gamma[l] (shared, or gamma[l,i] per
            qubit) and eta[l], then the driver with beta[l] (or beta[l,i]).
"""

from __future__ import annotations

import numpy as np

from aqfm_prescreen.aqfm import observables_batch, z_signs


def apply_1q(psi, U, q, p):
    """apply 2x2 gates U to qubit q.  psi (B, 2^p); U (2, 2) or (B, 2, 2)."""
    B = psi.shape[0]
    U = np.broadcast_to(U, (B, 2, 2))
    a = psi.reshape(B, 1 << (p - 1 - q), 2, 1 << q)
    a0, a1 = a[:, :, 0, :], a[:, :, 1, :]
    u = lambda i, j: U[:, i, j][:, None, None]
    out = np.stack([u(0, 0) * a0 + u(0, 1) * a1, u(1, 0) * a0 + u(1, 1) * a1], axis=2)
    return out.reshape(B, -1)


def ry(t):
    t = np.asarray(t, float)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def rz(t):
    t = np.asarray(t, float)
    z = np.zeros_like(t)
    return np.stack([np.stack([np.exp(-0.5j * t), z], -1), np.stack([z, np.exp(0.5j * t)], -1)], -2)


def xdrive(b):
    b = np.asarray(b, float)
    c, s = np.cos(b), 1j * np.sin(b)
    return np.stack([np.stack([c + 0j, s], -1), np.stack([s, c + 0j], -1)], -2)


def cnot_ring_perm(p):
    """index permutation of the CNOT ring 0->1, 1->2, ..., p-1->0 (applied in that order):
    new_psi = psi[perm]"""
    idx = np.arange(1 << p)
    perm = idx.copy()
    pairs = [(i, (i + 1) % p) for i in range(p)] if p > 2 else ([(0, 1), (1, 0)] if p == 2 else [])
    for c, t in pairs:                       # psi'[b] = psi[b ^ (bit_c(b) << t)]; compose in order
        src = idx ^ (((idx >> c) & 1) << t)
        perm = perm[src]
    return perm


def generic_forward(X, theta, reupload):
    """X (B, p); theta (L, p, 2).  Returns the final states (B, 2^p)."""
    B, p = X.shape
    psi = np.zeros((B, 1 << p), complex)
    psi[:, 0] = 1.0
    perm = cnot_ring_perm(p)
    for layer in range(theta.shape[0]):
        if layer == 0 or reupload:
            for q in range(p):
                psi = apply_1q(psi, ry(X[:, q]), q, p)
        for q in range(p):
            psi = apply_1q(psi, rz(theta[layer, q, 1]) @ ry(theta[layer, q, 0]), q, p)
        psi = psi[:, perm]
    return psi


def informed_forward(X, gamma, eta, beta):
    """X (B, p); gamma (L,) or (L, p); eta (L,); beta (L,) or (L, p).  Final states (B, 2^p)."""
    B, p = X.shape
    ZS = z_signs(p)                                           # (2^p, p)
    zz = 0.5 * (ZS.sum(1) ** 2 - p)                           # sum_{i<j} z_i z_j per basis state
    psi = np.full((B, 1 << p), 2.0 ** (-p / 2), complex)      # |+>^p
    for layer in range(len(eta)):
        g = np.broadcast_to(np.asarray(gamma[layer], float), (p,))
        b = np.broadcast_to(np.asarray(beta[layer], float), (p,))
        psi = psi * np.exp(1j * ((X * g) @ ZS.T + eta[layer] * zz))
        for q in range(p):
            psi = apply_1q(psi, xdrive(b[q]), q, p)
    return psi


def strang_schedule(J, tau, n_steps):
    """informed-circuit parameters that reproduce the map's Strang anneal EXACTLY:
    step k = diag(dt s/2) . driver(dt (1 - s)) . diag(dt s/2); consecutive half-diagonals merge
    (they commute), giving n_steps + 1 layers, the last with no driver."""
    dt = float(tau) / n_steps
    s = (np.arange(n_steps) + 0.5) / n_steps
    half = 0.5 * dt * s
    gamma = np.concatenate([[half[0]], half[1:] + half[:-1], [half[-1]]])
    beta = np.concatenate([dt * (1 - s), [0.0]])
    return gamma, J * gamma, beta


def readout(psi):
    """(B, 3p): <X_i>, <Y_i>, <Z_i> in the bank conventions"""
    p = int(np.log2(psi.shape[1]))
    return observables_batch(psi)[:, :3 * p]
