r"""
Which precision may a node-local solve use? Theorem 2 of the note, in closed form.

Notation as in Kremling and Speck (2021). For :math:`u' = Au` the collocation problem is
:math:`CU = U_0` with :math:`C = I - \Delta t\,(Q \otimes A)`, SDC is preconditioned with
:math:`P = I - \Delta t\,(Q_\Delta \otimes A)`, :math:`K = I - P^{-1}C` is its iteration matrix and
:math:`D = I - \Delta t\,(\mathrm{diag}(Q_\Delta) \otimes A)`. If SDC with exact solves needs
:math:`\bar k` iterations, solve errors change the residual after :math:`\bar k + a` iterations by a
sum over the iterations :math:`k` of :math:`C K^{\bar k + a - 1 - k} P^{-1}` applied to the solves'
residuals. Two models (:data:`MODELS`) bound them: a solve whose *result* is off by :math:`\eta`
times its maximum (residual :math:`D\varepsilon`), and a backward-stable solve in a format with unit
roundoff :math:`u`, whose *residual* is up to :math:`u\,\|D_j\|_\infty` times its maximum. If the
errors are independent and mean-zero, Hoeffding's inequality and a union bound give the threshold
of Theorem 2 (:func:`bound`), evaluated over the spectrum of the matrices of SDC for the Dahlquist
test equation. The feedback of the errors adds the rate condition on the forward error,
:math:`\phi\eta \le \rho\,(\rho^{-a/(\bar k + a)} - 1)`, with :math:`\phi = \kappa \approx \max_j
\|D_j\|_\infty` for a backward-stable solve. Since the bound is a sum over the iterations, it also
gives a format per iteration (:func:`schedule`). Everything can be computed from the exact run or
from the live data of a previous time step (:func:`record_run`, :func:`analyse`).
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


def record_run(spec, problem_class=None, problem_params=None, u0=None, t0=0.0, restol=-1, maxiter=60):
    """
    Run one step and record what the bounds need: residual history, each node's correction per
    iteration, the sweeper, the problem and the end value.

    With the defaults it is the exact run of ``spec``; any other run -- inexact solves, another
    initial value, stopped at ``restol`` -- is the live data of a time step. The corrections are read
    off the sweeper, not off the solve, which works on a right-hand side scaled to unit size (the
    correction-solve contract).
    """
    deltas = []

    class record(Hooks):
        def post_sweep(self, step, level_number):
            super().post_sweep(step, level_number)
            deltas.append([float(np.max(np.abs(np.asarray(d)))) for d in step.levels[0].sweep._deltas])

    description = {
        'problem_class': problem_class or spec['problem_class'],
        'problem_params': spec['problem_params'] if problem_params is None else problem_params,
        'sweeper_class': delta_implicit,
        'sweeper_params': spec['sweeper_params'],
        'level_params': {'restol': restol, 'dt': spec['dt']},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(1, {'logger_level': 40, 'hook_class': record}, description)
    level = controller.MS[0].levels[0]
    uend, stats = controller.run(u0=level.prob.u_exact(t0) if u0 is None else u0, t0=t0, Tend=t0 + spec['dt'])
    history = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    return {'history': history, 'deltas': np.array(deltas), 'sweep': level.sweep, 'prob': level.prob, 'uend': uend}


def exact_run(spec, maxiter=60):
    """The exact run of ``spec``: every node solve at working precision."""
    run = record_run(spec, maxiter=maxiter)
    return run['history'], run['deltas'], run['sweep'], run['prob']


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
        J = jacobian(prob).tocsc()
        lo, hi = (eigsh(J, k=1, which=which, return_eigenvectors=False)[0] for which in ('SA', 'LA'))
        return dt * np.linspace(lo, hi, 4000), 'max', 1.0, J.shape[0]
    return None


MODELS = ('result', 'residual')
"""How a solve errs (Assumption 1 of the note). ``result``: its result is off by up to eta times its
maximum -- a result rounded to a format, as :func:`.inexact.inexact` emulates. ``residual``: it solves
a nearby system, its residual is up to u times ||D_j|| times its maximum -- a backward-stable solve
in a format with unit roundoff u, or a Krylov solve stopped at a relative residual u."""


def jacobian(prob):
    """The (frozen) Jacobian of the right-hand side at the initial value."""
    if isinstance(prob, vanderpol_delta):
        x, y = prob.u0
        return np.array([[0.0, 1.0], [-2 * prob.mu * x * y - 1, prob.mu * (1 - x**2)]])
    if isinstance(prob, allencahn_delta):
        return prob.A + sp.diags(prob.reaction_prime(np.asarray(prob.u_exact(0.0)).ravel()))
    if hasattr(prob, 'A'):
        return prob.A
    raise NotImplementedError(f'no Jacobian for {type(prob).__name__}')


def node_norms(prob, dt, QD):
    """||I - dt q_jj J||_inf for every node j: the residual scale of a backward-stable node solve."""
    J = jacobian(prob)
    rows = (lambda B: np.asarray(abs(B).sum(1)).ravel()) if sp.issparse(J) else (lambda B: np.abs(B).sum(1))
    return np.array(
        [
            rows(sp.eye(J.shape[0]) - dt * q * J if sp.issparse(J) else np.eye(len(J)) - dt * q * J).max()
            for q in np.diag(QD)
        ]
    )


def contributions(spec, deltas, last, sweep, prob, model='residual'):
    """
    The contribution of each iteration k = 0..last to Psi^2, row by row.

    Returns
    -------
    tuple
        S with S[k, i] = sum_(j,n) |G^(k)_(i,(j,n))|^2 w_j^2, with G^(k) = C K^(last-k) P^-1 D and
        w_j = |delta^(k)_j| for the ``result`` model, G^(k) = C K^(last-k) P^-1 and
        w_j = ||D_j|| |delta^(k)_j| for the ``residual`` model (rows i are the nodes when the spectrum
        is known, else all unknowns), and the number of unknowns N.
    """
    if model not in MODELS:
        raise ValueError(f'model must be one of {MODELS}')
    Q, QD = sweep.coll.Qmat[1:, 1:], sweep.QI[1:, 1:]
    M = len(Q)
    weights = deltas if model == 'result' else deltas * node_norms(prob, spec['dt'], QD)
    known = spectrum(prob, spec['dt'])
    if known is not None:
        zs, reduce, c, N = known
        G2 = np.zeros((last + 1, M, M))  # |G_mj|^2 reduced over the spectrum, n = 0..last
        for z in zs:
            C, Kz, G = sdc_matrices(z, Q, QD)
            if model == 'residual':
                G = G @ np.diag(1 / (1 - z * np.diag(QD)))  # P^-1 D D^-1
            for n in range(last + 1):
                value = np.abs(C @ G) ** 2
                G2[n] = np.maximum(G2[n], value) if reduce == 'max' else G2[n] + value / len(zs)
                G = Kz @ G
        return np.array([c * G2[last - k] @ weights[k] ** 2 for k in range(last + 1)]), N

    # small or non-normal: the finite-dimensional formula, with the Jacobian at the initial value
    J = jacobian(prob)
    N = len(J)
    eye = np.eye(N * M)
    C = eye - spec['dt'] * np.kron(Q, J)
    Pinv = np.linalg.inv(eye - spec['dt'] * np.kron(QD, J))
    Kz = eye - Pinv @ C
    G = Pinv if model == 'residual' else Pinv @ (eye - spec['dt'] * np.kron(np.diag(np.diag(QD)), J))
    S = np.zeros((last + 1, N * M))
    for k in range(last, -1, -1):  # G = K^(last-k) P^-1 [D]
        S[k] = np.abs(C @ G) ** 2 @ np.repeat(weights[k] ** 2, N)
        G = Kz @ G
    return S, N


def analyse(spec, run=None, extra=1, p=CONFIDENCE, model='residual'):
    """
    Everything the bounds need, from the exact run or from the live data of a previous step.

    A live run stops at ``restol``, so it may lack the correction and residual of the iterations
    after its last; they are extrapolated with its contraction rate.

    Returns
    -------
    dict
        ``kbar``, ``rho``, the contributions ``S``, the ``margin`` (restol minus the residual after
        kbar + extra iterations, divided by c_p) and ``phi``, the factor between the bound on the
        solve's relative forward error and eta: 1 for the ``result`` model, kappa = max_j ||D_j||
        for the ``residual`` model (assuming ||D_j^-1|| <= 1, as for dissipative problems).
    """
    run = run or record_run(spec)
    history, deltas = list(run['history']), np.array(run['deltas'])
    kbar = iterations(history, spec['restol'])
    rho = contraction(history, spec['restol'])
    last = kbar + extra - 1
    while len(history) <= last:
        history.append(history[-1] * rho)
    while len(deltas) <= last:
        deltas = np.vstack([deltas, deltas[-1] * rho])
    S, N = contributions(spec, deltas, last, run['sweep'], run['prob'], model)
    c_p = np.sqrt(2 * np.log(2 * deltas.shape[1] * N / p))
    phi = 1.0 if model == 'result' else float(node_norms(run['prob'], spec['dt'], run['sweep'].QI[1:, 1:]).max())
    return {'kbar': kbar, 'rho': rho, 'S': S, 'margin': (spec['restol'] - history[last]) / c_p, 'phi': phi}


def bound(spec, p=CONFIDENCE, extra=1, model='residual', run=None):
    """
    The largest admissible eta (``result``) or unit roundoff (``residual``) for at most ``extra``
    additional iterations: Theorem 2 and the rate condition.

    Returns
    -------
    dict
        ``theorem`` (Theorem 2), ``rate`` (the rate condition on the forward error, divided by phi),
        ``kappa`` (phi) and ``eta_star``, the smaller of the two bounds.
    """
    a = analyse(spec, run, extra, p, model)
    theorem = a['margin'] / np.sqrt(a['S'].sum(0).max())
    rho, kbar = a['rho'], a['kbar']
    rate = rho * (rho ** (-extra / (kbar + extra)) - 1) / a['phi']
    return {'theorem': theorem, 'rate': rate, 'kappa': a['phi'], 'eta_star': min(theorem, rate)}


def schedule(spec, p=CONFIDENCE, formats=('fp16', 'fp32', 'fp64'), model='residual', run=None):
    """
    A format per iteration, cheapest first, such that at most one additional iteration is needed.

    Greedy: start every iteration in the coarsest format and refine the one with the largest
    contribution until Theorem 2 and the rate condition, prod_k (rho + phi u_k) <= rho^kbar, both
    hold. bf16 is left out by default: it costs as much as fp16 and is less accurate, and the
    corrections are scaled to unit size, so its range does not help.

    Returns
    -------
    list
        One format name per iteration 0..kbar.
    """
    a = analyse(spec, run, 1, p, model)
    S, rho, kbar = a['S'], a['rho'], a['kbar']
    level = np.zeros(kbar + 1, dtype=int)
    while True:
        u = np.array([FORMATS[formats[i]] for i in level])
        noise_ok = (u[:, None] ** 2 * S).sum(0).max() <= a['margin'] ** 2
        if noise_ok and np.prod(rho + a['phi'] * u) <= rho**kbar:
            return [formats[i] for i in level]
        score = u**2 * S.max(1) if not noise_ok else u.copy()
        score[level == len(formats) - 1] = -1
        if score.max() < 0:  # nothing left to refine: the finest format everywhere is the best there is
            return [formats[-1]] * (kbar + 1)
        level[int(np.argmax(score))] += 1


def kappa(spec):
    """max_j ||I - dt q_jj J||_inf: known before any solve (one iteration builds the sweeper and problem)."""
    run = record_run(spec, maxiter=1)
    return float(node_norms(run['prob'], spec['dt'], run['sweep'].QI[1:, 1:]).max())
