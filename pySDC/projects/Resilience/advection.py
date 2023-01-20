# script to run a simple advection problem

from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import get_sorted
import numpy as np
from pySDC.projects.Resilience.hook import LogData, hook_collection
from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults


def plot_embedded(stats, ax):
    u = get_sorted(stats, type='u', recomputed=False)
    uold = get_sorted(stats, type='uold', recomputed=False)
    t = [me[0] for me in u]
    e_em = get_sorted(stats, type='error_embedded_estimate', recomputed=False)
    e_em_semi_glob = [abs(u[i][1] - uold[i][1]) for i in range(len(u))]
    ax.plot(t, e_em_semi_glob, label=r'$\|u^{\left(k-1\right)}-u^{\left(k\right)}\|$')
    ax.plot([me[0] for me in e_em], [me[1] for me in e_em], linestyle='--', label=r'$\epsilon$')
    ax.set_xlabel(r'$t$')
    ax.legend(frameon=False)


def run_advection(
    custom_description=None,
    num_procs=1,
    Tend=2e-1,
    hook_class=LogData,
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

    problem_params = {'freq': 2, 'nvars': 2**9, 'c': 1.0, 'type': 'backward', 'order': 5, 'bc': 'periodic'}

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advectionNd  # pass problem class
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
        rnd_args = {
            'iteration': 5,
        }
        args = {
            'time': 1e-1,
            'target': 0,
        }
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.projects.Resilience.hook import LogUold

    adaptivity_params = dict()
    adaptivity_params['e_tol'] = 1e-8

    convergence_controllers = dict()
    convergence_controllers[Adaptivity] = adaptivity_params

    description = dict()
    description['convergence_controllers'] = convergence_controllers

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plot_embedded(run_advection(description, 1, hook_class=LogUold)[0], axs[0])
    plot_embedded(run_advection(description, 4, hook_class=LogUold)[0], axs[1])
    axs[0].set_title('1 process')
    axs[1].set_title('4 processes')
    fig.tight_layout()
    plt.show()
