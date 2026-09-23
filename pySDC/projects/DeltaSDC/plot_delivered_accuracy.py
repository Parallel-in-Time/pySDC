r"""
Plot for a talk: what the SDC iteration actually asks of its node-local solver.

The solver is a black box specified by a single number -- the relative accuracy :math:`\eta` of
the *correction* it returns. How it got there (a low-precision factorisation with iterative
refinement, a loose Krylov tolerance, a couple of multigrid cycles) is invisible to the iteration,
so it is swept directly here with :class:`~pySDC.projects.DeltaSDC.tests.inexact_problem.heat_inexact`,
which solves exactly and then spoils the answer by a random relative :math:`\eta`.

Left panel
    Residual per iteration for a few :math:`\eta`. Every curve contracts at the same rate and
    reaches the same floor; a less accurate solve only shifts the curve right.
Right panel
    Iterations to a fixed tolerance against :math:`\eta`, with the unit roundoff of each IEEE
    format marked. The star is a real ``float16`` node-local solve, which lands on the sweep at
    its own unit roundoff -- the model and the emulation agree.

Run as ``python -m pySDC.projects.DeltaSDC.plot_delivered_accuracy``; writes
``delivered_accuracy`` as PDF and PNG.
"""

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.projects.DeltaSDC.problems import heat_delta
from pySDC.projects.DeltaSDC.run_demo import HEAT_PARAMS, SWEEPER_PARAMS
from pySDC.projects.DeltaSDC.tests.inexact_problem import heat_inexact

TOL = 1e-11
"""Residual tolerance the iteration counts are measured against."""

MAXITER = 40
SEEDS = (0, 1, 2)
"""The perturbation is random, so each eta is run several times and the median reported."""

# unit roundoff, not eps: the largest relative error of correctly rounding into the format
FORMATS = [('fp16', 2.0**-11), ('fp32', 2.0**-24), ('fp64', 2.0**-53)]

ETAS = sorted({10.0**-k for k in range(2, 17)} | {3e-3, 3e-4, 3e-5, 3e-6} | {u for _, u in FORMATS})

# one hue, monotone lightness: eta is an ordered magnitude, not an identity
CURVES = [
    (1e-2, '#86b6ef'),
    (1e-3, '#5598e7'),
    (1e-4, '#2a78d6'),
    (1e-6, '#1c5cab'),
    (None, '#0d366b'),
]

GRID = {'linewidth': 0.6, 'alpha': 0.4}


def residuals(problem_class, problem_extra):
    """
    Residual after every iteration of a single heat-equation step.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_extra : dict
        Overrides for :data:`~pySDC.projects.DeltaSDC.run_demo.HEAT_PARAMS`.

    Returns
    -------
    list
        One residual per iteration.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': dict(HEAT_PARAMS, **problem_extra),
        'sweeper_class': delta_implicit,
        'sweeper_params': dict(SWEEPER_PARAMS, linear_implicit=True),
        'level_params': {'restol': -1, 'dt': 1e-1},
        'step_params': {'maxiter': MAXITER},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    return [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]


def iterations(history):
    """
    First iteration at which the residual falls below :data:`TOL`.

    Parameters
    ----------
    history : list
        Residual per iteration.

    Returns
    -------
    int or None
        The iteration, or ``None`` if the tolerance is never reached.
    """
    return next((i + 1 for i, value in enumerate(history) if value < TOL), None)


def main():
    """
    Run the sweep, draw both panels and write them as PDF and PNG.

    Returns
    -------
    dict
        ``etas``, the median iteration count per eta, and the real float16 run's count.
    """
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 12})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for eta, color in CURVES:
        history = residuals(heat_inexact, {} if eta is None else {'eta': eta, 'seed': 0})
        left.semilogy(
            range(1, len(history) + 1),
            history,
            color=color,
            linewidth=2,
            marker='o',
            markersize=5,
            label='exact solve' if eta is None else rf'$\eta = 10^{{{int(np.log10(eta))}}}$',
        )
    left.set_xlabel('SDC iteration')
    left.set_ylabel('residual')
    left.set_xlim(0.5, 20.5)
    left.set_xticks([1] + list(range(4, 21, 4)))
    left.grid(True, **GRID)
    left.set_axisbelow(True)
    left.legend(frameon=False, loc='lower left')
    left.set_title('a less accurate solve only shifts the curve')

    counts = [int(np.median([iterations(residuals(heat_inexact, {'eta': e, 'seed': s})) for s in SEEDS])) for e in ETAS]
    exact = iterations(residuals(heat_inexact, {}))
    genuine = iterations(residuals(heat_delta, {'solve_precision': np.float16}))

    right.semilogx(ETAS, counts, color='#2a78d6', linewidth=2, marker='o', markersize=7, zorder=3)
    right.axhline(exact, color='#6b6b6b', linewidth=1.2, linestyle='--', zorder=1)
    right.annotate(
        f'exact solve: {exact} iterations',
        xy=(ETAS[2], exact),
        xytext=(0, 8),
        textcoords='offset points',
        color='#6b6b6b',
        fontsize=12,
    )
    for name, roundoff in FORMATS:
        right.axvline(roundoff, color='#9a9a9a', linewidth=1, linestyle=':', zorder=1)
        right.annotate(
            name,
            xy=(roundoff, 1.0),
            xycoords=('data', 'axes fraction'),
            xytext=(3, -14),
            textcoords='offset points',
            color='#6b6b6b',
            fontsize=13,
        )
    right.plot(
        [2.0**-11],
        [genuine],
        marker='*',
        markersize=18,
        color='#eb6834',
        linestyle='none',
        zorder=4,
    )
    right.annotate(
        'real float16 solve',
        xy=(2.0**-11, genuine),
        xytext=(-12, -22),
        textcoords='offset points',
        color='#eb6834',
        fontsize=13,
        ha='right',
    )
    right.set_xlabel(r'relative accuracy $\eta$ delivered by the solver')
    right.set_ylabel(f'iterations to a residual of {TOL:.0e}')
    right.set_ylim(min(counts + [exact]) - 1.5, max(counts + [exact]) + 1.5)
    right.grid(True, **GRID)
    right.set_axisbelow(True)
    right.set_title('and fp16 is nearly free')

    for axes in (left, right):
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)
    fig.tight_layout()
    for suffix in ('pdf', 'png'):
        fig.savefig(f'delivered_accuracy.{suffix}', dpi=200)
    plt.close(fig)
    return {'etas': ETAS, 'iterations': counts, 'exact': exact, 'float16': genuine}


if __name__ == '__main__':
    result = main()
    print(f"exact solve: {result['exact']} iterations, real float16 solve: {result['float16']}")
    for eta, count in zip(result['etas'], result['iterations'], strict=True):
        print(f'  eta {eta:.1e}: {count}')
