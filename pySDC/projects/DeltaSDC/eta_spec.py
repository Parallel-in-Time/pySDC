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
    from first-order perturbation theory, with two mechanisms, and the smaller of the two:

    *noise*. A solve error made in sweep :math:`j` alone leaves a residual :math:`T_j\,\eta` at
    iteration :math:`K + 1`, measured by spoiling that sweep only. Independent errors superpose, so
    errors in every sweep leave :math:`\eta\,(\sum_j T_j^2)^{1/2}`, and :math:`\eta^*_\mathrm{noise} =
    \mathrm{tol}\,/\,(\sum_j T_j^2)^{1/2}`. Nothing is linearised and nothing is decomposed, so it holds
    for nonlinear problems and opaque solvers alike; it costs :math:`K + 1` probing runs.

    *rate*. The error is relative to the correction, so it feeds back: each sweep contracts by about
    :math:`\rho + \eta` instead of :math:`\rho`, and an iteration is lost once :math:`(\rho +
    \eta)^{K+1}` no longer beats :math:`\rho^K`: :math:`\eta^*_\mathrm{rate} = \rho\,(\rho^{-1/(K+1)} -
    1)`, from the exact run alone. It is 1e-2 to 3e-2 throughout, so it binds only where the noise
    does not -- non-stiff problems without a spectral gap -- and never for a format finer than
    ``bfloat16``.

The measured contraction rate alone does not predict :math:`\eta^*`: the solve's error excites every
mode, while the solution converges in its own, and a stiff residual sees a solve error amplified by
roughly :math:`\Delta t\,\|J\|`. The noise mechanism follows both.

Run as ``python -m pySDC.projects.DeltaSDC.eta_spec``; prints the table and writes ``data/eta_spec``
as PDF and PNG.
"""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.DeltaSDC.inexact import inexact
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


def case(family, label, problem_class, problem_params, dt, tol, M=3, QI='LU', sweeper=None, eta_level=None):
    """One configuration of the sweep, as a dict."""
    return {
        'family': family,
        'label': label,
        'problem_class': problem_class,
        'problem_params': problem_params,
        'dt': dt,
        'tol': tol,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': M, 'QI': QI, **(sweeper or {})},
        'eta_level': eta_level,
    }


def cases():
    """
    The configurations: step size, nodes and preconditioner varied on heat, and the other problems
    at a few step sizes each. ``eta_level`` marks an MLSDC run and which of its solves is spoiled.
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
            case('heat MLSDC', f'MLSDC {level} solve', heat_delta, HEAT, 1e-1, 1e-11, sweeper=LINEAR, eta_level=level)
            for level in ['fine', 'coarse']
        ],
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
    if spec['eta_level']:
        params['nvars'] = [params['nvars'], (params['nvars'] - 1) // 2]
        params['eta'] = [eta, None] if spec['eta_level'] == 'fine' else [None, eta]
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'iorder': 4, 'rorder': 2}
        description['base_transfer_class'] = delta_transfer
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=spec['dt'])
    return [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]


def iterations(history, tol):
    """Iterations until the residual first falls below ``tol``; infinity if it never does."""
    return next((k + 1 for k, value in enumerate(history) if value < tol), np.inf)


def contraction(history, tol):
    """Geometric mean of the ratio of successive residuals, over the iterations above ``tol``."""
    above = [value for value in history if value >= tol]
    return (above[-1] / above[0]) ** (1 / (len(above) - 1)) if len(above) > 1 else np.nan


PROBE = 1e-2
"""Relative size of the single-sweep perturbations that measure the transfer coefficients."""


def predict(spec, seeds=SEEDS):
    """
    eta* from first-order perturbation theory: the smaller of the noise and the rate bound.

    Returns
    -------
    dict
        ``noise`` and ``rate``, the two bounds, and ``eta_star``, the smaller.
    """
    history = residuals(spec)
    K = iterations(history, spec['tol'])
    nodes = spec['sweeper_params']['num_nodes']
    transfer = []
    for sweep in range(1, K + 2):
        calls = set(range((sweep - 1) * nodes, sweep * nodes))
        hit = np.median([residuals(spec, PROBE, seed, K + 1, calls)[K] for seed in seeds])
        transfer.append(max(float(hit) - history[K], 0.0) / PROBE)
    noise = spec['tol'] / np.sqrt(np.sum(np.square(transfer)))
    rho = contraction(history, spec['tol'])
    rate = rho * (rho ** (-1 / (K + 1)) - 1)
    return {'noise': noise, 'rate': rate, 'eta_star': min(noise, rate)}


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
    exact = iterations(exact_history, spec['tol'])
    counts = [float(np.median([iterations(residuals(spec, eta, seed), spec['tol']) for seed in seeds])) for eta in etas]
    # from the most accurate eta upwards: eta* is the last one before the first that costs more
    eta_star = None
    for eta, count in sorted(zip(etas, counts, strict=True)):
        if count > exact + 1:
            break
        eta_star = eta
    return {
        'exact': exact,
        'rho': contraction(exact_history, spec['tol']),
        'counts': counts,
        'eta_star': eta_star,
        'predicted': None if spec['eta_level'] else predict(spec, seeds),
    }


# one hue per family, from the project's palette
COLORS = {
    'heat': '#2a78d6',
    'heat MLSDC': '#1c5cab',
    'advection': '#eb6834',
    'Allen-Cahn': '#4a3aa7',
    'Van der Pol': '#2e9c6a',
}


def figure(results, etas=ETAS):
    """Iterations over eta for every configuration, and eta* over the contraction rate."""
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'legend.fontsize': 11})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    for spec, result in results:
        extra = [count - result['exact'] for count in result['counts']]
        left.semilogx(etas, extra, color=COLORS[spec['family']], linewidth=1.2, alpha=0.8)
    left.axhline(1, color='#6b6b6b', linewidth=1, linestyle='--')
    left.set_xlabel(r'relative accuracy $\eta$ delivered by the solver')
    left.set_ylabel('extra iterations')
    left.set_ylim(-0.5, 10)
    left.set_title('every configuration: flat, then a knee')

    for family, color in COLORS.items():
        points = [
            (r['predicted']['eta_star'], r['eta_star']) for s, r in results if s['family'] == family and r['predicted']
        ]
        if points:
            predicted, measured = zip(*points, strict=True)
            right.loglog(predicted, measured, 'o', color=color, markersize=8, label=family)
    line = np.array([1e-9, 1e1])
    right.loglog(line, line, color='#9a9a9a', linewidth=1, linestyle=':')
    right.fill_between(line, line / 3, line * 3, color='#9a9a9a', alpha=0.15, linewidth=0, label='within a factor 3')
    right.set_xlabel(r'$\eta^*$ predicted by perturbation theory')
    right.set_ylabel(r'$\eta^*$ measured: largest $\eta$ at $\leq 1$ extra iteration')
    right.legend(frameon=False, loc='lower right')
    right.set_title(r'the accuracy a solve needs, predicted')

    for axes in (left, right):
        axes.grid(True, linewidth=0.6, alpha=0.4)
        axes.set_axisbelow(True)
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)
    fig.tight_layout()
    Path('data').mkdir(exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(f'data/eta_spec.{suffix}', dpi=200)
    plt.close(fig)


def main(specs=None, etas=ETAS, seeds=SEEDS):
    """
    Measure every configuration, print the table and draw the figure.

    Returns
    -------
    list
        ``(spec, result)`` per configuration.
    """
    results = [(spec, measure(spec, etas, seeds)) for spec in (specs or cases())]
    print(f"{'configuration':>30} | {'iter':>4} {'rho':>8} | {'eta*':>8} {'noise':>8} {'rate':>8}")
    print('-' * 76)
    for spec, result in results:
        eta_star = '-' if result['eta_star'] is None else f"{result['eta_star']:.0e}"
        p = result['predicted']
        bounds = f"{'-':>8} {'-':>8}" if p is None else f"{p['noise']:>8.1e} {p['rate']:>8.1e}"
        print(f"{spec['label']:>30} | {result['exact']:>4} {result['rho']:>8.1e} | {eta_star:>8} {bounds}")
    figure(results, etas)
    return results


if __name__ == '__main__':
    main()
