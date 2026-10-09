r"""
When does a low-precision node-local solve cost nothing? Theorem B of the note, in closed form.

Delta-form SDC is preconditioned Richardson iteration on the collocation system :math:`Cu = b`,
:math:`C = I - \Delta t\,Q \otimes A`, with the sweep :math:`P = I - \Delta t\,Q_\Delta \otimes A` as
preconditioner. An error :math:`\varepsilon` in the node-local solves enters the residual as

.. math::
    r^{k+1} = E\,r^k - F\,\varepsilon^k, \qquad E = I - C P^{-1}, \qquad F = C P^{-1} D,

with :math:`D` the block diagonal of :math:`P`. If the errors are independent, mean-zero and bounded by
:math:`\eta` times the node's correction, Hoeffding's inequality and a union bound give, with
probability at least :math:`1 - p`, an extra residual after iteration :math:`K + 1` of at most
:math:`\eta\,\lambda_p\,\Psi`, :math:`\lambda_p = \sqrt{2 \log(2MN/p)}`, where

.. math::
    \Psi^2 = \max_i \sum_{j=0}^{K} \sum_l \big|(E^{K-j} F)_{il}\big|^2\,\|\delta_{j,l}\|_\infty^2 .

For a normal operator with delocalised eigenvectors the sum over unknowns is an integral over the
spectral measure :math:`\mu` of :math:`\Delta t\,A`, of :math:`M \times M` functions of the scalar
:math:`z` -- the test-equation matrices :math:`E(z)`, :math:`F(z)`:

.. math::
    \Psi^2 \le c_V \max_m \sum_{j=0}^{K} \sum_n \|\delta_{j,n}\|_\infty^2
    \int \big|[E(z)^{K-j} F(z)]_{mn}\big|^2\,d\mu(z),

:math:`c_V = N \max |V_{x\lambda}|^2`, 1 for a Fourier and 2 for a sine basis. :math:`\mu` is known in
closed form for the finite-difference operators, so nothing is decomposed. The run costs at most one
iteration more as long as :math:`\eta \le \eta^*_B = (\mathrm{tol} - \|r^{K+1}\|)/(\lambda_p \Psi)`.
That is first order; the feedback of the error into later corrections adds the rate condition
:math:`\eta \le \rho\,(\rho^{-1/(K+1)} - 1)`, which binds only for non-stiff problems.
"""

import numpy as np

from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta, vanderpol_delta

CONFIDENCE = 0.05
"""p: the bound holds with probability at least 1 - p."""


def iterations(history, tol):
    """Iterations until the residual first falls below ``tol``; infinity if it never does."""
    return next((k + 1 for k, value in enumerate(history) if value < tol), np.inf)


def contraction(history, tol):
    """Geometric mean of the ratio of successive residuals, over the iterations above ``tol``."""
    above = [value for value in history if value >= tol]
    return (above[-1] / above[0]) ** (1 / (len(above) - 1)) if len(above) > 1 else np.nan


def test_matrices(z, Q, QD):
    """The scalar test-equation matrices: residual map E(z), error injection F(z)."""
    eye = np.eye(len(Q))
    P, C = eye - z * QD, eye - z * Q
    CPinv = C @ np.linalg.inv(P)
    return eye - CPinv, CPinv @ np.diag(np.diag(P))


def exact_run(spec, maxiter=60):
    """
    The exact run: residual history, each node's correction per sweep, the sweeper and the problem.

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


def spectral_measure(prob, dt):
    """
    The spectral measure of dt times the implicit operator, in closed form: points, weights summing
    to one, the delocalisation constant c_V and the number of unknowns. ``None`` for a problem without one.
    """
    if isinstance(prob, heat_delta):  # 1D, Dirichlet, second order: a sine basis
        N = prob.A.shape[0]
        k = np.arange(1, N + 1)
        z = -dt * prob.nu * 4 * (N + 1) ** 2 * np.sin(k * np.pi / (2 * (N + 1))) ** 2
        return z, np.full(N, 1 / N), 2 * N / (N + 1), N
    if isinstance(prob, advectionNd):  # 1D, periodic, fourth-order centred: a Fourier basis
        N = prob.A.shape[0]
        theta = 2 * np.pi * np.arange(N) / N
        z = -dt * prob.c * 1j * N * (8 * np.sin(theta) - np.sin(2 * theta)) / 6
        return z, np.full(N, 1 / N), 1.0, N
    if isinstance(prob, allencahn_delta):
        # the periodic Laplacian shifted by the reaction's derivative in the bulk, its median: away
        # from the thin interface the state sits in a well, where the derivative is -2/eps^2, and
        # almost all unknowns are there. The interface's few positive modes are left out -- an
        # approximation, the Jacobian is not diagonalised by the Laplacian's basis
        n = prob.nvars[0]
        one = -4 * n**2 * np.sin(np.pi * np.arange(n) / n) ** 2
        bulk = float(np.median(prob.reaction_prime(np.asarray(prob.u_exact(0.0)).ravel())))
        z = dt * ((one[:, None] + one[None, :]).ravel() + bulk)
        return z, np.full(n * n, 1 / n**2), 1.0, n * n
    return None


def psi(spec, deltas, K, sweep, prob):
    """
    Psi of Theorem B: from the spectral measure where there is one, else exactly from the full matrices.

    Returns
    -------
    tuple
        Psi and the number of unknowns N.
    """
    Q, QD = sweep.coll.Qmat[1:, 1:], sweep.QI[1:, 1:]
    M = len(Q)
    measure = spectral_measure(prob, spec['dt'])
    if measure is not None:
        zs, weights, c_V, N = measure
        integral = np.zeros((K + 1, M, M))  # integral of |[E^n F]_mn|^2 d mu, for n = 0..K
        for z, w in zip(zs, weights, strict=True):
            E, F = test_matrices(z, Q, QD)
            G = F
            for n in range(K + 1):
                integral[n] += w * np.abs(G) ** 2
                G = E @ G
        total = sum(integral[K - j] @ deltas[j] ** 2 for j in range(K + 1))
        return np.sqrt(c_V * total.max()), N

    # small or non-normal: the finite-dimensional formula, with the Jacobian at the initial value
    J = jacobian(prob)
    N = len(J)
    eye = np.eye(N * M)
    P = eye - spec['dt'] * np.kron(QD, J)
    C = eye - spec['dt'] * np.kron(Q, J)
    D = eye - spec['dt'] * np.kron(np.diag(np.diag(QD)), J)
    CPinv = C @ np.linalg.inv(P)
    E, G = eye - CPinv, CPinv @ D
    total = np.zeros(N * M)
    for j in range(K, -1, -1):  # G = E^(K-j) F
        total += np.abs(G) ** 2 @ np.repeat(deltas[j] ** 2, N)
        G = E @ G
    return np.sqrt(total.max()), N


def jacobian(prob):
    """Van der Pol's Jacobian at the initial value."""
    if isinstance(prob, vanderpol_delta):
        x, y = prob.u0
        return np.array([[0.0, 1.0], [-2 * prob.mu * x * y - 1, prob.mu * (1 - x**2)]])
    raise NotImplementedError(f'no spectral measure or Jacobian for {type(prob).__name__}')


def bound(spec, p=CONFIDENCE):
    """
    eta* from Theorem B and from the rate condition, and the smaller of the two.

    Returns
    -------
    dict
        ``theorem`` (eta*_B), ``rate`` and ``eta_star``.
    """
    history, deltas, sweep, prob = exact_run(spec)
    K = iterations(history, spec['tol'])
    value, N = psi(spec, deltas, K, sweep, prob)
    M = deltas.shape[1]
    theorem = (spec['tol'] - history[K]) / (np.sqrt(2 * np.log(2 * M * N / p)) * value)
    rho = contraction(history, spec['tol'])
    rate = rho * (rho ** (-1 / (K + 1)) - 1)
    return {'theorem': theorem, 'rate': rate, 'eta_star': min(theorem, rate)}
