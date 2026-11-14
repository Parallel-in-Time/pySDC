"""
The monolithic flow around the cylinder of
:mod:`pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_monolithic_FEniCS`, with the Firedrake port of the
problem class. Drag and lift are logged per step by
:class:`~pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake.LogLiftDrag`.
"""

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake import LogLiftDrag
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import (
    NavierStokes2DMonolithicFiredrake,
)


def setup(t0=3.125e-4):
    """
    The parameters of the FEniCS version, with the residual tolerance in the dual norm the Firedrake data types use.

    Returns:
        description and controller_params dictionaries, and t0
    """
    description = {
        'problem_class': NavierStokes2DMonolithicFiredrake,
        'problem_params': {'nu': 0.001, 'order': 2},
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QI': 'LU'},
        'level_params': {'restol': 1e-10, 'dt': 1 / 100},
        'step_params': {'maxiter': 10},
    }
    return description, {'logger_level': 20, 'hook_class': [LogLiftDrag]}, t0


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


if __name__ == '__main__':
    description, controller_params, t0 = setup()
    run_simulation(description, controller_params, t0, Tend=8.0)
