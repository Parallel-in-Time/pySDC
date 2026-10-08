r"""
Figures: SDC absorbs a half-precision node-local solve.

One figure per problem -- the 1D heat equation (linear, direct solve) and 2D Allen-Cahn (nonlinear,
Newton-CG) -- each showing the residual per iteration of a delta-form run with an fp64 solve, the
same run with a half-precision solve, and a control that has to stall:

heat
    The linear route: the sweeper hands the solve its right-hand side at unit size (the
    correction-solve contract on :class:`~pySDC.core.problem.Problem`), so the half-precision solve
    needs nothing of its own. The control is stock SDC with the same solve, which hands it the
    state instead of a correction.
Allen-Cahn
    The nonlinear route, where the scaled unknown would enter :math:`f`, so the solver scales
    internally. The control is the same solve with that switched off: the correction falls below
    half precision's range and is flushed to zero.

Both runs go to a fixed iteration count rather than to a tolerance, so the tail is a floor and not
whatever stopped the run.

Run as ``python -m pySDC.projects.DeltaSDC.plot_mixed_precision``; writes
``data/mixed_precision_heat`` and ``data/mixed_precision_allencahn`` as PDF and PNG. The project's
tests run it, so the README and the website show what the current code produces.
"""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta
from pySDC.projects.DeltaSDC.run_demo import BASE_PARAMS, HEAT_PARAMS, SWEEPER_PARAMS

# categorical slots 1, 2 and 7 of the project's palette, validated for colour-vision deficiency
STYLES = [('#2a78d6', 'o', '-'), ('#eb6834', 's', '--'), ('#4a3aa7', '^', ':')]

FP16 = {'solve_precision': np.float16}

FIGURES = [
    # name, problem class, base parameters, extra sweeper parameters, dt, iterations, title, and the
    # series: the fp64 reference, the half-precision run, and last the control, as
    # (label, problem parameters, sweeper class)
    (
        'heat',
        heat_delta,
        HEAT_PARAMS,
        {'linear_implicit': True},
        1e-1,
        18,
        '1D heat equation, linear',
        [
            ('fp64 solve', {}, delta_implicit),
            ('fp16 solve', FP16, delta_implicit),
            ('fp16 solve, stock SDC', FP16, generic_implicit),
        ],
    ),
    (
        'allencahn',
        allencahn_delta,
        BASE_PARAMS,
        {},
        4e-3,
        16,
        r'2D Allen-Cahn, nonlinear',
        [
            ('fp64 solve', {}, delta_implicit),
            ('fp16 solve, normalized', FP16, delta_implicit),
            ('fp16 solve, naive', dict(FP16, normalize=False), delta_implicit),
        ],
    ),
]


def residuals(problem_class, problem_params, sweeper_class, sweeper_extra, dt, maxiter):
    """
    Residual after every iteration of a single step.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for it.
    sweeper_class : type
        Sweeper class to use.
    sweeper_extra : dict
        Overrides for :data:`~pySDC.projects.DeltaSDC.run_demo.SWEEPER_PARAMS`.
    dt : float
        Step size.
    maxiter : int
        Number of iterations to run.

    Returns
    -------
    list
        One residual per iteration.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': dict(SWEEPER_PARAMS, **sweeper_extra),
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=dt)
    return [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]


def figure(name, problem_class, base_params, sweeper_extra, dt, maxiter, title, series):
    """
    Draw one problem's convergence plot and write it as PDF and PNG.

    Parameters
    ----------
    name : str
        File name stem.
    problem_class : type
        Problem class to integrate.
    base_params : dict
        Its parameters, before the per-series overrides.
    sweeper_extra : dict
        Overrides for :data:`~pySDC.projects.DeltaSDC.run_demo.SWEEPER_PARAMS`.
    dt : float
        Step size.
    maxiter : int
        Number of iterations to run.
    title : str
        Axes title.
    series : list
        ``(label, problem parameters, sweeper class)`` per curve: reference, half precision, control.

    Returns
    -------
    dict
        The residual history per label.
    """
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 14})
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    curves = {}
    for (label, extra, sweeper_class), (color, marker, linestyle) in zip(series, STYLES, strict=True):
        curves[label] = residuals(problem_class, dict(base_params, **extra), sweeper_class, sweeper_extra, dt, maxiter)
        ax.semilogy(
            range(1, len(curves[label]) + 1),
            curves[label],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=8,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel('SDC iteration')
    ax.set_ylabel('residual')
    ax.set_xticks([1] + list(range(3, maxiter + 1, 3)))
    ax.set_xlim(0.5, maxiter + 0.5)
    ax.grid(True, which='major', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()

    Path('data').mkdir(exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(f'data/mixed_precision_{name}.{suffix}', dpi=200)
    plt.close(fig)
    return curves


def main():
    """
    Draw every figure.

    Returns
    -------
    dict
        The residual histories, keyed by figure name and then by label.
    """
    return {spec[0]: figure(*spec) for spec in FIGURES}


if __name__ == '__main__':
    for name, histories in main().items():
        print(name)
        for label, history in histories.items():
            print(f'  {label:>28}: floor {min(history):.1e} after {len(history)} iterations')
