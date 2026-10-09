r"""
How accurate does a node-local solve have to be? Theorem 2 of the note, in closed form.

Notation as in Kremling and Speck (2021). For :math:`u' = Au` the collocation problem is
:math:`CU = U_0` with :math:`C = I - \Delta t\,(Q \otimes A)`, SDC is preconditioned with
:math:`P = I - \Delta t\,(Q_\Delta \otimes A)`, and :math:`K = I - P^{-1}C` is its iteration matrix. If
SDC with exact solves needs :math:`\bar k` iterations, a node-local solve that returns its correction
with an error :math:`\varepsilon^{(k)}` changes the residual :math:`r^{(k)} = U_0 - CU^{(k)}` after
:math:`\bar k + m` iterations by

.. math::
    \sum_{k=0}^{\bar k + m - 1} C K^{\bar k + m - 1 - k} P^{-1} D\,\varepsilon^{(k)},
    \qquad D = I - \Delta t\,(\mathrm{diag}(Q_\Delta) \otimes A).

If the errors are independent, mean-zero and bounded by :math:`\eta` times the node's correction
:math:`\delta^{(k)}_j`, Hoeffding's inequality and a union bound bound this, with probability at least
:math:`1 - p`, by :math:`\eta\,c_p\,\Psi`, :math:`c_p = \sqrt{2 \log(2MN/p)}`. :math:`\Psi^2` is a sum of
one contribution per iteration, each a mean (constant coefficients, normal :math:`A`) or a maximum
(Allen-Cahn: frozen symmetric Jacobian) over the spectrum of the matrices of SDC for the Dahlquist
test equation. At most :math:`m` additional iterations are needed as long as
:math:`\eta \le \eta^* = (\mathrm{restol} - \|r^{(\bar k + m)}\|_\infty)/(c_p \Psi)`; the feedback of
the errors adds the heuristic rate condition :math:`\eta \le \rho\,(\rho^{-m/(\bar k + m)} - 1)`.
Since the contributions are per iteration, they also give a format per iteration (:func:`schedule`).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.projects.DeltaSDC.inexact import FORMATS
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta, vanderpol_delta

CONFIDENCE = 0.05
"""p: the bound holds with probability at least 1 - p."""


def iterations(history, restol):
    """Iterations until the residual first falls below ``restol``; infinity if it never does."""
    return next((k + 1 for k, value in enumerate(history) if value < restol), np.inf)


def contraction(history, restol):
    """Geometric mean of the ratio of successive residuals, over the iterations above ``restol``."""
    above = [value for value in history if value >= restol]
    return (above[-1] / above[0]) ** (1 / (len(above) - 1)) if len(above) > 1 else np.nan


def sdc_matrices(z, Q, QD):
    """SDC for the Dahlquist test equation, z = dt * lambda: C, the iteration matrix K and P^-1 D."""
    eye = np.eye(len(Q))
    C, P = eye - z * Q, eye - z * QD
    Pinv = np.linalg.inv(P)
    return C, eye - Pinv @ C, Pinv @ np.diag(np.diag(P))


def exact_run(spec, maxiter=60):
    """
    The exact run: residual history, each node's correction per iteration, the sweeper and the problem.

    The corrections are read off the sweeper, not off the solve, which works on a right-hand side
    scaled to unit size (the correction-solve contract).
    """
    deltas = []

    class record(Hooks):
        def post_sweep(self, step, level_number):
            super().post_sweep(step, level_number)
            deltas.append([float(np.max(np.abs(np.asarray(d)))) for d in step.levels[0].sweep._deltas])

    description = {
        'problem_class': spec['problem_class'],
        'problem_params': spec['problem_params'],
        'sweeper_class': delta_implicit,
        'sweeper_params': spec['sweeper_params'],
        'level_params': {'restol': -1, 'dt': spec['dt']},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(1, {'logger_level': 40, 'hook_class': record}, description)
    level = controller.MS[0].levels[0]
    _, stats = controller.run(u0=level.prob.u_exact(0.0), t0=0.0, Tend=spec['dt'])
    history = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    return history, np.array(deltas), level.sweep, level.prob


def spectrum(prob, dt):
    """
    dt times the eigenvalues of the (frozen) Jacobian, how to reduce over them, the constant and the
    number of unknowns. ``None`` for a problem without them, which is then treated with full matrices.

    Constant-coefficient problems use the mean over the eigenvalues, with c_V from the eigenvectors
    (eq. (10) of the note). Allen-Cahn's Jacobian at u_0 is symmetric but not diagonalised by the
    Fourier basis; it uses the maximum over the interval spanned by its extreme eigenvalues, which
    bounds every row (Corollary 2 of the note).
    """
    if isinstance(prob, heat_delta):  # 1D, Dirichlet, second order: a sine basis
        N = prob.A.shape[0]
        ell = np.arange(1, N + 1)
        z = -dt * prob.nu * 4 * (N + 1) ** 2 * np.sin(ell * np.pi / (2 * (N + 1))) ** 2
        return z, 'mean', 2 * N / (N + 1), N
    if isinstance(prob, advectionNd):  # 1D, periodic, fourth-order centred: a Fourier basis
        N = prob.A.shape[0]
        theta = 2 * np.pi * np.arange(N) / N
        return -dt * prob.c * 1j * N * (8 * np.sin(theta) - np.sin(2 * theta)) / 6, 'mean', 1.0, N
    if isinstance(prob, allencahn_delta):
        u0 = np.asarray(prob.u_exact(0.0)).ravel()
        J = (prob.A + sp.diags(prob.reaction_prime(u0))).tocsc()
        lo, hi = (eigsh(J, k=1, which=which, return_eigenvectors=False)[0] for which in ('SA', 'LA'))
        return dt * np.linspace(lo, hi, 4000), 'max', 1.0, J.shape[0]
    return None


def contributions(spec, deltas, last, sweep, prob):
    """
    The contribution of each iteration k = 0..last to Psi^2, row by row.

    Returns
    -------
    tuple
        S with S[k, i] = sum_(j,n) |[C K^(last-k) P^-1 D]_(i,(j,n))|^2 |delta^(k)_j|^2 (rows i are
        the nodes when the spectrum is known, else all unknowns), and the number of unknowns N.
    """
    Q, QD = sweep.coll.Qmat[1:, 1:], sweep.QI[1:, 1:]
    M = len(Q)
    known = spectrum(prob, spec['dt'])
    if known is not None:
        zs, reduce, c, N = known
        G2 = np.zeros((last + 1, M, M))  # |[C K^n P^-1 D]_mj|^2 reduced over the spectrum, n = 0..last
        for z in zs:
            C, Kz, G = sdc_matrices(z, Q, QD)
            for n in range(last + 1):
                value = np.abs(C @ G) ** 2
                G2[n] = np.maximum(G2[n], value) if reduce == 'max' else G2[n] + value / len(zs)
                G = Kz @ G
        return np.array([c * G2[last - k] @ deltas[k] ** 2 for k in range(last + 1)]), N

    # small or non-normal: the finite-dimensional formula, with the Jacobian at the initial value
    J = jacobian(prob)
    N = len(J)
    eye = np.eye(N * M)
    C = eye - spec['dt'] * np.kron(Q, J)
    Pinv = np.linalg.inv(eye - spec['dt'] * np.kron(QD, J))
    Kz, G = eye - Pinv @ C, Pinv @ (eye - spec['dt'] * np.kron(np.diag(np.diag(QD)), J))
    S = np.zeros((last + 1, N * M))
    for k in range(last, -1, -1):  # G = K^(last-k) P^-1 D
        S[k] = np.abs(C @ G) ** 2 @ np.repeat(deltas[k] ** 2, N)
        G = Kz @ G
    return S, N


def jacobian(prob):
    """Van der Pol's Jacobian at the initial value."""
    if isinstance(prob, vanderpol_delta):
        x, y = prob.u0
        return np.array([[0.0, 1.0], [-2 * prob.mu * x * y - 1, prob.mu * (1 - x**2)]])
    raise NotImplementedError(f'no spectrum or Jacobian for {type(prob).__name__}')


def _setup(spec, extra, p):
    """The exact run, its iteration count and rate, the contributions for ``extra`` iterations and c_p."""
    history, deltas, sweep, prob = exact_run(spec)
    kbar = iterations(history, spec['restol'])
    last = kbar + extra - 1
    S, N = contributions(spec, deltas, last, sweep, prob)
    c_p = np.sqrt(2 * np.log(2 * deltas.shape[1] * N / p))
    margin = (spec['restol'] - history[last]) / c_p
    return kbar, contraction(history, spec['restol']), S, margin


def bound(spec, p=CONFIDENCE, extra=1):
    """
    eta* from Theorem 2 and from the rate condition for at most ``extra`` additional iterations.

    Returns
    -------
    dict
        ``theorem`` (eta* of Theorem 2), ``rate`` and ``eta_star``, the smaller of the two.
    """
    kbar, rho, S, margin = _setup(spec, extra, p)
    theorem = margin / np.sqrt(S.sum(0).max())
    rate = rho * (rho ** (-extra / (kbar + extra)) - 1)
    return {'theorem': theorem, 'rate': rate, 'eta_star': min(theorem, rate)}


def schedule(spec, p=CONFIDENCE, formats=('fp16', 'fp32', 'fp64')):
    """
    A format per iteration, cheapest first, such that at most one additional iteration is needed.

    Greedy: start every iteration in the coarsest format and refine the one with the largest
    contribution until Theorem 2 and the rate condition both hold. bf16 is left out by default: it
    costs as much as fp16 and is less accurate, and the corrections are scaled to unit size, so its
    range does not help.

    Returns
    -------
    list
        One format name per iteration 0..kbar.
    """
    kbar, rho, S, margin = _setup(spec, 1, p)
    level = np.zeros(kbar + 1, dtype=int)
    while True:
        u = np.array([FORMATS[formats[i]] for i in level])
        noise_ok = (u[:, None] ** 2 * S).sum(0).max() <= margin**2
        if noise_ok and np.prod(rho + u) <= rho**kbar:
            return [formats[i] for i in level]
        score = u**2 * S.max(1) if not noise_ok else u.copy()
        score[level == len(formats) - 1] = -1
        level[int(np.argmax(score))] += 1
