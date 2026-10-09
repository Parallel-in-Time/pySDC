r"""
How accurate does a node-local solve have to be? Theorem 2 of the note, in closed form.

Notation as in Kremling and Speck (2021). For :math:`u' = Au` the collocation problem is
:math:`CU = U_0` with :math:`C = I - \Delta t\,(Q \otimes A)`, SDC is preconditioned with
:math:`P = I - \Delta t\,(Q_\Delta \otimes A)`, and :math:`K = I - P^{-1}C` is its iteration matrix. A
node-local solve that returns its correction with an error :math:`\varepsilon^{(k)}` changes the residual
:math:`r^{(k)} = U_0 - CU^{(k)}` after iteration :math:`K + 1` by

.. math::
    \sum_{k=0}^{K} C K^{K-k} P^{-1} D\,\varepsilon^{(k)}, \qquad D = I - \Delta t\,(\mathrm{diag}(Q_\Delta) \otimes A).

If the errors are independent, mean-zero and bounded by :math:`\eta` times the node's correction
:math:`\delta^{(k)}_j`, Hoeffding's inequality and a union bound bound this, with probability at least
:math:`1 - p`, by :math:`\eta\,c_p\,\Psi`, :math:`c_p = \sqrt{2 \log(2MN/p)}`, with

.. math::
    \Psi^2 \le c_V \max_m \sum_{k=0}^{K} \sum_{j=1}^{M} \|\delta^{(k)}_j\|_\infty^2\,
    \frac{1}{N} \sum_{\ell} \big|[C K^{K-k} P^{-1} D]_{mj}(\Delta t \lambda_\ell)\big|^2

for a normal :math:`A` with eigenvalues :math:`\lambda_\ell`: the matrices are those of SDC for the
Dahlquist test equation, and :math:`c_V = 1` for a Fourier, :math:`2N/(N + 1)` for a sine basis. The
eigenvalues of the finite-difference operators are known in closed form, so nothing is diagonalised.
The run costs at most one iteration more as long as
:math:`\eta \le \eta^* = (\mathrm{restol} - \|r^{(K+1)}\|_\infty)/(c_p \Psi)`. The feedback of the errors
into later corrections adds the rate condition :math:`\eta \le \rho\,(\rho^{-1/(K+1)} - 1)`, which
binds only for non-stiff problems.
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


def eigenvalues(prob, dt):
    """
    dt times the eigenvalues of the implicit operator, in closed form, with the constant c_V and the
    number of unknowns. ``None`` for a problem without them.
    """
    if isinstance(prob, heat_delta):  # 1D, Dirichlet, second order: a sine basis
        N = prob.A.shape[0]
        ell = np.arange(1, N + 1)
        return -dt * prob.nu * 4 * (N + 1) ** 2 * np.sin(ell * np.pi / (2 * (N + 1))) ** 2, 2 * N / (N + 1), N
    if isinstance(prob, advectionNd):  # 1D, periodic, fourth-order centred: a Fourier basis
        N = prob.A.shape[0]
        theta = 2 * np.pi * np.arange(N) / N
        return -dt * prob.c * 1j * N * (8 * np.sin(theta) - np.sin(2 * theta)) / 6, 1.0, N
    if isinstance(prob, allencahn_delta):
        # the periodic Laplacian shifted by the reaction's derivative in the bulk, its median: away
        # from the thin interface the state sits in a well, where the derivative is -2/eps^2, and
        # almost all unknowns are there. The interface's few positive modes are left out -- an
        # approximation, the Jacobian is not diagonalised by the Fourier basis
        n = prob.nvars[0]
        one = -4 * n**2 * np.sin(np.pi * np.arange(n) / n) ** 2
        bulk = float(np.median(prob.reaction_prime(np.asarray(prob.u_exact(0.0)).ravel())))
        return dt * ((one[:, None] + one[None, :]).ravel() + bulk), 1.0, n * n
    return None


def psi(spec, deltas, K, sweep, prob):
    """
    Psi of Theorem 2: over the eigenvalues where they are known, else exactly from the full matrices.

    Returns
    -------
    tuple
        Psi and the number of unknowns N.
    """
    Q, QD = sweep.coll.Qmat[1:, 1:], sweep.QI[1:, 1:]
    M = len(Q)
    known = eigenvalues(prob, spec['dt'])
    if known is not None:
        zs, c_V, N = known
        mean = np.zeros((K + 1, M, M))  # mean over the eigenvalues of |[C K^n P^-1 D]_mj|^2, n = 0..K
        for z in zs:
            C, Kz, G = sdc_matrices(z, Q, QD)
            for n in range(K + 1):
                mean[n] += np.abs(C @ G) ** 2 / len(zs)
                G = Kz @ G
        total = sum(mean[K - k] @ deltas[k] ** 2 for k in range(K + 1))
        return np.sqrt(c_V * total.max()), N

    # small or non-normal: the finite-dimensional formula, with the Jacobian at the initial value
    J = jacobian(prob)
    N = len(J)
    eye = np.eye(N * M)
    C = eye - spec['dt'] * np.kron(Q, J)
    Pinv = np.linalg.inv(eye - spec['dt'] * np.kron(QD, J))
    Kz, G = eye - Pinv @ C, Pinv @ (eye - spec['dt'] * np.kron(np.diag(np.diag(QD)), J))
    total = np.zeros(N * M)
    for k in range(K, -1, -1):  # G = K^(K-k) P^-1 D
        total += np.abs(C @ G) ** 2 @ np.repeat(deltas[k] ** 2, N)
        G = Kz @ G
    return np.sqrt(total.max()), N


def jacobian(prob):
    """Van der Pol's Jacobian at the initial value."""
    if isinstance(prob, vanderpol_delta):
        x, y = prob.u0
        return np.array([[0.0, 1.0], [-2 * prob.mu * x * y - 1, prob.mu * (1 - x**2)]])
    raise NotImplementedError(f'no eigenvalues or Jacobian for {type(prob).__name__}')


def bound(spec, p=CONFIDENCE):
    """
    eta* from Theorem 2 and from the rate condition, and the smaller of the two.

    Returns
    -------
    dict
        ``theorem`` (eta* of Theorem 2), ``rate`` and ``eta_star``.
    """
    history, deltas, sweep, prob = exact_run(spec)
    K = iterations(history, spec['restol'])
    value, N = psi(spec, deltas, K, sweep, prob)
    M = deltas.shape[1]
    theorem = (spec['restol'] - history[K]) / (np.sqrt(2 * np.log(2 * M * N / p)) * value)
    rho = contraction(history, spec['restol'])
    rate = rho * (rho ** (-1 / (K + 1)) - 1)
    return {'theorem': theorem, 'rate': rate, 'eta_star': min(theorem, rate)}
