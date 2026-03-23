import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types

from pySDC.playgrounds.time_dep_BCs.heat_eq_time_dep_BCs import Heat1DTimeDependentBCs
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.hooks.plotting import PlotPostStep


def run_heat(dt=1e-1, Tend=4, plotting=False):
    level_params = {}
    level_params['dt'] = dt

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {}

    step_params = {}
    step_params['maxiter'] = 4

    controller_params = {}
    controller_params['logger_level'] = 15
    if plotting:
        controller_params['hook_class'] = PlotPostStep

    description = {}
    description['problem_class'] = Heat1DTimeDependentBCs
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    t0 = 0.0

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, controller


if __name__ == '__main__':
    run_heat(plotting=True)
    plt.show()
