r"""
The propagators real-time TDDFT codes use, on the same problem class and work counters as the SDC runs.

``strang``, ``triple_jump``
    Split-operator :math:`T \mid V_{nl} \mid V_{loc}`, Strang (order 2) and its triple-jump composition (order 4).
    The projector exponential is exact (rank-few update), the local potential uses the density after the first half
    steps. The kinetic half steps of consecutive steps are merged in the cost.
``aetrs``
    Approximate enforced time-reversal symmetry with 4th-order Taylor exponentials, the default in Octopus. The
    potential at the end of the step is extrapolated linearly.
``cfm4``
    Commutator-free Magnus 4 with Lanczos exponentials (orbital by orbital, stopping tolerance ``tol``). The
    potentials at the two Gauss points are extrapolated (cubic) from the previous steps.
``crank_nicolson``
    Crank-Nicolson with the midpoint potential extrapolated, solved by GMRES preconditioned with the kinetic inverse.

Cost: orbital FFT round trips (``fft`` counter), the same unit as for SDC.
"""

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spla

from pySDC.projects.TDDFT.problem import KohnShamMolecule1D


class PotentialHistory:
    """Local potentials of the last steps, for the extrapolating propagators"""

    def __init__(self):
        self.V, self.t = [], []

    def push(self, V, t):
        self.V.insert(0, V)
        self.t.insert(0, t)
        del self.V[4:], self.t[4:]

    def extrapolate(self, t, order):
        n = min(order + 1, len(self.V))
        ts, Vs = self.t[:n], self.V[:n]
        return sum(np.prod([(t - ts[j]) / (ts[i] - ts[j]) for j in range(n) if j != i]) * Vs[i] for i in range(n))


def kinetic_exp(P, psi, tau):
    P.work_counters['fft'].niter += psi.shape[0]
    return np.fft.ifft(np.exp(-1j * tau * P.T) * np.fft.fft(psi, axis=-1), axis=-1)


def projector_exp(P, psi, tau):
    r"""Exact :math:`e^{-i\tau V_{nl}}\psi = \psi + B^T (e^{-i\tau E_{kb} G} - I) G^{-1} (B\psi\,dx)`, G the Gram matrix"""
    P.work_counters['proj'].niter += psi.shape[0]
    B, dx = P.beta, P.dx
    G = B @ B.T * dx
    W = (sl.expm(-1j * tau * P.E_kb * G) - np.eye(len(G))) @ np.linalg.inv(G)
    return psi + ((psi @ B.T) * dx) @ W.T @ B


def strang_step(P, psi, t, dt):
    psi = kinetic_exp(P, psi, dt / 2)
    P.work_counters['fft'].niter -= psi.shape[0]  # merged with the second half of the previous step
    psi = projector_exp(P, psi, dt / 2)
    psi = np.exp(-1j * dt * P.potential(psi, t + dt / 2)) * psi
    psi = projector_exp(P, psi, dt / 2)
    return kinetic_exp(P, psi, dt / 2)


def triple_jump_step(P, psi, t, dt):
    g1 = 1 / (2 - 2 ** (1 / 3))
    g2 = 1 - 2 * g1
    for g, s in [(g1, 0), (g2, g1), (g1, g1 + g2)]:
        psi = strang_step(P, psi, t + s * dt, g * dt)
    return psi


def taylor_exp(P, psi, tau, V, order=4):
    r""":math:`e^{-i\tau H}\psi` by a truncated Taylor series (H with the local potential V)"""
    v, acc = psi, psi.copy()
    for k in range(1, order + 1):
        v = (-1j * tau / k) * P.apply_H(v, None, V=V)
        acc = acc + v
    return acc


def lanczos_exp(P, psi, tau, V, tol=1e-10, mmax=60):
    r""":math:`e^{-i\tau H}\psi` by Lanczos, orbital by orbital, stopped when the last coefficient is below tol"""
    out = np.empty_like(psi)
    for j in range(psi.shape[0]):
        q0 = psi[j : j + 1]
        beta0 = np.linalg.norm(q0)
        Q, alpha, betas = [q0 / beta0], [], []
        for _ in range(mmax):
            w = P.apply_H(Q[-1], None, V=V)
            a = np.vdot(Q[-1], w)
            w = w - a * Q[-1] - (betas[-1] * Q[-2] if betas else 0)
            alpha.append(a)
            b = np.linalg.norm(w)
            Tm = np.diag(alpha) + np.diag(betas, 1) + np.diag(betas, -1)
            ev, U = np.linalg.eig(Tm)
            c = U @ (np.exp(-1j * tau * ev) * np.linalg.solve(U, np.eye(len(alpha))[:, 0]))
            if abs(c[-1]) * b < tol or b < 1e-14:
                break
            betas.append(b)
            Q.append(w / b)
        out[j] = beta0 * sum(ci * qi[0] for ci, qi in zip(c, Q, strict=True))
    return out


def aetrs_step(P, psi, t, dt, hist, exp=taylor_exp):
    V0 = P.potential(psi, t)
    hist.push(V0, t)
    if len(hist.V) == 1:  # first step: predictor for the end potential
        V1 = P.potential(exp(P, psi, dt, V0), t + dt)
    else:
        V1 = hist.extrapolate(t + dt, 1)
    return exp(P, exp(P, psi, dt / 2, V0), dt / 2, V1)


def cfm4_step(P, psi, t, dt, hist, tol=1e-10):
    exp = lambda P, psi, tau, V: lanczos_exp(P, psi, tau, V, tol=tol)  # noqa: E731
    V0 = P.potential(psi, t)
    hist.push(V0, t)
    if len(hist.V) < 4:  # start-up with AETRS steps until the extrapolation has its history
        hist.V.pop(0), hist.t.pop(0)
        return aetrs_step(P, psi, t, dt, hist, exp)
    c1, c2 = 0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6
    a1, a2 = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
    V1, V2 = hist.extrapolate(t + c1 * dt, 3), hist.extrapolate(t + c2 * dt, 3)
    # exp(-i dt (a H1 + b H2)) with a + b = 1/2 is exp(-i dt/2 (T + 2a V1 + 2b V2))
    psi = exp(P, psi, dt / 2, 2 * (a2 * V1 + a1 * V2))
    return exp(P, psi, dt / 2, 2 * (a1 * V1 + a2 * V2))


def crank_nicolson_step(P, psi, t, dt, hist, tol=1e-11):
    V0 = P.potential(psi, t)
    hist.push(V0, t)
    Vm = hist.extrapolate(t + dt / 2, 1) if len(hist.V) > 1 else V0
    shape, n = psi.shape, psi.size

    def H(v):
        return P.apply_H(v.reshape(shape), None, V=Vm).ravel()

    def prec(v):
        P.work_counters['fft'].niter += shape[0]
        return np.fft.ifft(np.fft.fft(v.reshape(shape), axis=-1) / (1 + 0.5j * dt * P.T), axis=-1).ravel()

    A = spla.LinearOperator((n, n), matvec=lambda v: v + 0.5j * dt * H(v), dtype=complex)
    M = spla.LinearOperator((n, n), matvec=prec, dtype=complex)
    x, _ = spla.gmres(A, psi.ravel() - 0.5j * dt * H(psi.ravel()), x0=psi.ravel(), M=M, rtol=tol, atol=0, restart=50)
    return x.reshape(shape)


def run_propagator(method, dt, Tend, dt_out, problem_params=None, tol=1e-10):
    """
    Propagate the ground state to Tend with fixed steps and record the dipole and norm every dt_out.

    Returns:
        dict: times ``t``, dipole ``d``, norm ``n``, and ``cost`` (orbital FFT round trips)
    """
    P = KohnShamMolecule1D(**(problem_params or {}))
    psi = np.asarray(P.ground_state())
    hist = PotentialHistory()
    step = {
        'strang': lambda psi, t: strang_step(P, psi, t, dt),
        'triple_jump': lambda psi, t: triple_jump_step(P, psi, t, dt),
        'aetrs': lambda psi, t: aetrs_step(P, psi, t, dt, hist),
        'cfm4': lambda psi, t: cfm4_step(P, psi, t, dt, hist, tol=tol),
        'crank_nicolson': lambda psi, t: crank_nicolson_step(P, psi, t, dt, hist),
    }[method]
    if method in ['strang', 'triple_jump']:
        P.work_counters['fft'].niter += P.norb  # the very first half step is not merged
    rec = dict(t=[0.0], d=[P.dipole(psi)], n=[P.norm(psi)])
    every = int(round(dt_out / dt))
    for k in range(int(round(Tend / dt))):
        psi = step(psi, k * dt)
        if (k + 1) % every == 0:
            rec['t'].append((k + 1) * dt)
            rec['d'].append(P.dipole(psi))
            rec['n'].append(P.norm(psi))
        if not np.all(np.isfinite(psi)):
            break
    return dict(**{key: np.array(v) for key, v in rec.items()}, cost=P.work_counters['fft'].niter)
