"""
The heat equation of :mod:`pySDC.projects.StroemungsRaum.run_heat_equation_FEniCS` with the Firedrake port of the
problem class. The solutions go to a Firedrake ``CheckpointFile``, which ``plotting/plots_heat_equation_Firedrake.py``
reads.
"""

import json
import os
from pathlib import Path

import firedrake as fd

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.projects.StroemungsRaum.problem_classes.HeatEquation_2D_Firedrake import Heat2DFiredrake

DEFAULT_PATH = f'{os.path.dirname(__file__)}/data/heat_equation/'


def setup(t0=0.0):
    """
    The parameters of the FEniCS version, with its residual tolerance 1e-12 converted to the dual norm the Firedrake
    data types use: on this 64 x 64 mesh with P2 elements, the dual norm of a residual is about 1.2e4 times the norm
    FEniCS takes (the L2 norm of the function with the residual's entries as coefficients), so 1e-12 becomes about 1e-8.

    Returns:
        description and controller_params dictionaries, and t0
    """
    description = {
        'problem_class': Heat2DFiredrake,
        'problem_params': {'nu': 0.1, 'c_nvars': 64, 'order': 2, 'c': 0.0},
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2},
        'level_params': {'restol': 1e-8, 'dt': 0.2},
        'step_params': {'maxiter': 20},
    }
    controller_params = {'logger_level': 20, 'hook_class': LogSolution}
    return description, controller_params, t0


def run_simulation(description, controller_params, t0, Tend):
    """
    Run the simulation.

    Returns:
        The problem, the statistics and the relative error at `Tend`
    """
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    return P, stats, abs(uex - uend) / abs(uex)


def run_postprocessing(description, problem, stats, Tend, path=DEFAULT_PATH):
    """
    Write the parameters to a JSON file and the numerical and exact solutions of every step, as ``un`` and ``ux`` with
    the step number as index, to ``heat_equation_Firedrake_Temperature.h5``.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    parameters = {**description['problem_params'], **description['level_params'], 'Tend': Tend}
    with open(f'{path}/heat_equation_Firedrake_parameters.json', 'w') as f:
        json.dump(parameters, f)

    mesh = problem.V.mesh()
    with fd.CheckpointFile(f'{path}/heat_equation_Firedrake_Temperature.h5', 'w', comm=mesh.comm) as file:
        file.save_mesh(mesh)
        for i, (time, un) in enumerate(get_sorted(stats, type='u', sortby='time')):
            file.save_function(un.functionspace, idx=i, name='un')
            file.save_function(problem.u_exact(time).functionspace, idx=i, name='ux')


if __name__ == '__main__':
    Tend = 1.0
    description, controller_params, t0 = setup()
    problem, stats, rel_err = run_simulation(description, controller_params, t0, Tend)
    print('The relative error at time ', Tend, 'is ', rel_err)
    run_postprocessing(description, problem, stats, Tend)
