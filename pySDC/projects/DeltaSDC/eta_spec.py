r"""
How accurate does a node-local solve have to be? Measured across problems, step sizes and methods.

For each configuration the solve is made deliberately inexact -- exact, then spoiled by a random
relative error :math:`\eta` (:func:`.inexact.inexact`) -- and the iterations to a fixed residual are
counted against :math:`\eta`. Two numbers come out:

``rho``
    the contraction rate of the run with exact solves: the geometric mean of the ratio of successive
    residuals while it converges;
``eta*``
    the largest :math:`\eta` at which the run, and every run with a more accurate solve, costs at
    most one iteration more than the exact one.

``eta* predicted``
    from :mod:`.precision_theory`, without running anything inexact: the
    extra residual independent, mean-zero solve errors leave, bounded with probability 95% -- in
    closed form over the operator's spectral measure, and the rate condition for the feedback of the
    errors into later corrections; the smaller of the two.
"""

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.projects.DeltaSDC.inexact import inexact
from pySDC.projects.DeltaSDC.precision_theory import bound, contraction, iterations
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta, vanderpol_delta

ETAS = [10.0 ** (-k / 2) for k in range(2, 21)]
"""1e-1 down to 1e-10, half a decade apart."""

SEEDS = (0, 1, 2)
"""The perturbation is random, so each eta is run several times and the median taken."""

HEAT = {'nvars': 127, 'nu': 1.0, 'freq': 2, 'bc': 'dirichlet-zero', 'solver_type': 'direct'}
ADVECTION = {'nvars': 128, 'c': 1.0, 'freq': 2, 'order': 4, 'solver_type': 'direct'}
ALLEN_CAHN = {
    'nvars': (64, 64),
    'eps': 0.04,
    'radius': 0.25,
    'newton_maxiter': 100,
    'newton_tol': 1e-14,
    'lin_tol': 1e-2,
    'lin_maxiter': 500,
}

LINEAR = {'linear_implicit': True}


def case(family, label, problem_class, problem_params, dt, restol, M=3, QI='LU', sweeper=None):
    """One configuration of the sweep, as a dict."""
    return {
        'family': family,
        'label': label,
        'problem_class': problem_class,
        'problem_params': problem_params,
        'dt': dt,
        'restol': restol,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': M, 'QI': QI, **(sweeper or {})},
    }


def cases():
    """
    The configurations: step size, nodes and preconditioner varied on heat, and the other problems
    at a few step sizes each.
    """
    dx = 1 / ADVECTION['nvars']
    return [
        *[
            case('heat', f'heat dt={dt:g}', heat_delta, HEAT, dt, 1e-11, sweeper=LINEAR)
            for dt in [1e-3, 1e-2, 1e-1, 1.0]
        ],
        *[case('heat', f'heat M={M}', heat_delta, HEAT, 1e-1, 1e-11, M=M, sweeper=LINEAR) for M in [2, 5]],
        *[case('heat', f'heat {QI}', heat_delta, HEAT, 1e-1, 1e-11, QI=QI, sweeper=LINEAR) for QI in ['MIN-SR-S']],
        *[
            case('advection', f'advection CFL={cfl:g}', advectionNd, ADVECTION, cfl * dx, 1e-11, sweeper=LINEAR)
            for cfl in [0.5, 2.0, 8.0]
        ],
        case(
            'advection',
            'advection CFL=2 MIN-SR-S',
            advectionNd,
            ADVECTION,
            2 * dx,
            1e-11,
            QI='MIN-SR-S',
            sweeper=LINEAR,
        ),
        *[
            case('Allen-Cahn', f'Allen-Cahn dt={dt:g}', allencahn_delta, ALLEN_CAHN, dt, 1e-10)
            for dt in [1e-3, 4e-3, 1.6e-2]
        ],
        case('Allen-Cahn', 'Allen-Cahn dt=0.004 MIN-SR-S', allencahn_delta, ALLEN_CAHN, 4e-3, 1e-10, QI='MIN-SR-S'),
        *[
            case('Van der Pol', f'Van der Pol mu={mu:g}', vanderpol_delta, {'mu': mu}, 0.05, 1e-11)
            for mu in [1.0, 10.0, 100.0]
        ],
        case('Van der Pol', 'Van der Pol mu=1 MIN-SR-NS', vanderpol_delta, {'mu': 1.0}, 0.05, 1e-11, QI='MIN-SR-NS'),
    ]


def residuals(spec, eta=None, seed=0, maxiter=60, calls=None):
    """
    Residual after every iteration of one step, with the solve spoiled by ``eta``.

    Returns
    -------
    list
        One residual per iteration.
    """
    params = dict(spec['problem_params'], eta=eta, seed=seed, calls=calls)
    description = {
        'problem_class': inexact(spec['problem_class']),
        'problem_params': params,
        'sweeper_class': delta_implicit,
        'sweeper_params': spec['sweeper_params'],
        'level_params': {'restol': -1, 'dt': spec['dt']},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=spec['dt'])
    return [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]


def measure(spec, etas=ETAS, seeds=SEEDS):
    """
    The exact run's iterations and contraction rate, the iterations per eta, and eta*.

    Returns
    -------
    dict
        ``exact``, ``rho``, ``counts`` (median iterations per eta, in the order of ``etas``) and
        ``eta_star`` (``None`` if even the most accurate eta tried costs more than one iteration).
    """
    exact_history = residuals(spec)
    exact = iterations(exact_history, spec['restol'])
    counts = [
        float(np.median([iterations(residuals(spec, eta, seed), spec['restol']) for seed in seeds])) for eta in etas
    ]
    # from the most accurate eta upwards: eta* is the last one before the first that costs more
    eta_star = None
    for eta, count in sorted(zip(etas, counts, strict=True)):
        if count > exact + 1:
            break
        eta_star = eta
    return {
        'exact': exact,
        'rho': contraction(exact_history, spec['restol']),
        'counts': counts,
        'eta_star': eta_star,
        'predicted': bound(spec, model='result'),
    }
