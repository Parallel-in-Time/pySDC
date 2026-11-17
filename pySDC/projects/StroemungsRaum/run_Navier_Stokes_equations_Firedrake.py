"""
The projection scheme for the flow around the cylinder (the DFG benchmark 2D-3), with
:class:`NavierStokes2DProjectionFiredrake` and its sweeper. Drag and lift are logged by
:class:`~pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake.LogLiftDrag`;
add :class:`~pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake.LogWriteSolutions` to the hooks to write the
solutions for ``plotting/plots_Navier_Stokes_projection_Firedrake.py``.
"""

import json
import os
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake import LogLiftDrag, LogWriteSolutions
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_projection_Firedrake import (
    NavierStokes2DProjectionFiredrake,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.projects.StroemungsRaum.convergence_controllers.prepare_step import PrepareStep

DEFAULT_PATH = f'{os.path.dirname(__file__)}/data/navier_stokes/'


def setup(t0=3.125e-4):
    """
    The parameters of the project's original study. The iteration stops when the increment between two iterations
    falls below `e_tol`.

    Returns:
        description and controller_params dictionaries, and t0
    """
    description = {
        'problem_class': NavierStokes2DProjectionFiredrake,
        'problem_params': {'nu': 0.001, 'order': 2},
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QE': 'PIC', 'QI': 'LU'},
        'level_params': {'e_tol': 1e-9, 'dt': 1 / 1600},
        'step_params': {'maxiter': 10},
        # the projection needs the start of every step
        'convergence_controllers': {PrepareStep: {}},
    }
    controller_params = {'logger_level': 20, 'hook_class': [LogLiftDrag]}
    return description, controller_params, t0


def run_simulation(description, controller_params, t0, Tend):
    """
    Run the simulation.

    Returns:
        The problem, the statistics and the solution at `Tend`
    """
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    return P, stats, uend


def run_postprocessing(description, stats, path=DEFAULT_PATH):
    """
    Write the parameters to a JSON file and the end time, drag and lift of every step to ``Lift_Drag_Coefficients.txt``.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    lift = get_sorted(stats, type='lift_post_step', sortby='time')
    drag = get_sorted(stats, type='drag_post_step', sortby='time')
    parameters = {
        **description['problem_params'],
        **description['level_params'],
        # the hook records the end time of each step
        't0': lift[0][0] - description['level_params']['dt'],
        'Tend': lift[-1][0],
        'timing': get_sorted(stats, type='timing_run', sortby='time')[0][1],
    }
    with open(f'{path}/Navier_Stokes_Firedrake_parameters.json', 'w') as f:
        json.dump(parameters, f)
    with open(f'{path}/Lift_Drag_Coefficients.txt', 'w') as f:
        for (time, d), (_, l) in zip(drag, lift, strict=True):
            f.write('%1.16f  %1.16f  %1.16f\n' % (time, d, l))


if __name__ == '__main__':
    description, controller_params, t0 = setup()
    controller_params['hook_class'].append(LogWriteSolutions)
    problem, stats, uend = run_simulation(description, controller_params, t0, Tend=0.005)
    run_postprocessing(description, stats)
