# script to run a simple heat problem

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.hook import log_error_estimates
import numpy as np


def plot_embedded(stats, ax):
    u = get_sorted(stats, type='u', recomputed=False)
    uold = get_sorted(stats, type='uold', recomputed=False)
    t = [get_sorted(stats, type='u', recomputed=False)[i][0] for i in range(len(u))]
    e_em = np.array(get_sorted(stats, type='e_embedded', recomputed=False))[:, 1]
    e_em_semi_glob = [abs(u[i][1] - uold[i][1]) for i in range(len(u))]
    ax.plot(t, e_em_semi_glob, label=r'$\|u^{\left(k-1\right)}-u^{\left(k\right)}\|$')
    ax.plot(t, e_em, linestyle='--', label=r'$\epsilon$')
    ax.set_xlabel(r'$t$')
    ax.legend(frameon=False)


def run_heat(
    custom_description=None,
    num_procs=1,
    Tend=2e-1,
    hook_class=log_error_estimates,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
):

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 0.05

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    problem_params = {
        'freq': 2,
        'nvars': 2**9,
        'nu': 1.0,
        'type': 'center',
        'order': 6,
        'bc': 'periodic',
        'direct_solver': True,
        'lintol': None,
        'liniter': None,
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_unforced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        raise NotImplementedError("The parameters have not been adapted to this equation yet!")
        controller.hooks.random_generator = fault_stuff['rng']
        controller.hooks.add_fault(
            rnd_args={'iteration': 5, **fault_stuff.get('rnd_params', {})},
            args={'time': 1e-1, 'target': 0, **fault_stuff.get('args', {})},
        )

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


if __name__ == '__main__':
    run_heat()
