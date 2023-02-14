# script to run a simple heat problem

from pySDC.implementations.problem_classes.LeakySuperconductor import LeakySuperconductor, LeakySuperconductorIMEX
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.hook import hook_collection, LogData
import numpy as np

import matplotlib.pyplot as plt
from pySDC.core.Errors import ConvergenceError


class live_plot(hooks):  # pragma no cover
    """
    This hook plots the solution and the non-linear part of the right hand side after every step. Keep in mind that using adaptivity will result in restarts, which is not marked in these plots. Prepare to see the temperature profile jumping back again after a restart.
    """

    def pre_run(self, step, level_number):  # pragma no cover
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))

    def post_step(self, step, level_number):  # pragma no cover
        for ax in self.axs:
            ax.cla()
        L = step.levels[level_number]
        self.axs[0].plot(L.prob.xv, L.u[-1])
        self.axs[0].axhline(L.prob.params.u_thresh, color='black')
        self.axs[1].plot(L.prob.xv, L.prob.eval_f_non_linear(L.u[-1], L.time))
        self.axs[0].set_ylim(0, 0.025)
        plt.pause(1e-9)


def run_leaky_superconductor(
    custom_description=None,
    num_procs=1,
    Tend=6e2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    imex=False,
    u0=None,
    t0=None,
):
    """
    Run a toy problem of a superconducting magnet with a temperature leak with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        custom_problem_params (dict): Overwrite presets
        imex (bool): Solve the problem IMEX or fully implicit
        u0 (dtype_u): Initial value
        t0 (float): Starting time

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'

    problem_params = {}

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
    description['problem_class'] = LeakySuperconductorIMEX if imex else LeakySuperconductor
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order if imex else generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0 if t0 is None else t0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        raise NotImplementedError("The parameters have not been adapted to this equation yet!")

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0) if u0 is None else u0

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError:
        stats = controller.return_stats()
    return stats, controller, Tend


def plot_solution(stats, controller):  # pragma no cover
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    u_ax = ax
    dt_ax = u_ax.twinx()

    u = get_sorted(stats, type='u', recomputed=False)
    u_ax.plot([me[0] for me in u], [max(me[1]) for me in u], label=r'$T$')

    dt = get_sorted(stats, type='dt', recomputed=False)
    dt_ax.plot([me[0] for me in dt], [me[1] for me in dt], color='black', ls='--')
    u_ax.plot([None], [None], color='black', ls='--', label=r'$\Delta t$')

    P = controller.MS[0].levels[0].prob
    u_ax.axhline(P.params.u_thresh, color='grey', ls='-.', label=r'$T_\mathrm{thresh}$')
    u_ax.axhline(P.params.u_max, color='grey', ls=':', label=r'$T_\mathrm{max}$')

    u_ax.legend()
    u_ax.set_xlabel(r'$t$')
    u_ax.set_ylabel(r'$T$')
    dt_ax.set_ylabel(r'$\Delta t$')


def compare_imex_full(plotting=False):
    """
    Compare the results of IMEX and fully implicit runs. For IMEX we need to limit the step size in order to achieve convergence, but for fully implicit, adaptivity can handle itself better.

    Args:
        plotting (bool): Plot the solution or not
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    res = {}

    dt_max = {True: 2e1, False: np.inf}
    custom_description = {}
    custom_description['problem_params'] = {
        'newton_tol': 1e-4,
        'newton_iter': 15,
    }
    custom_controller_params = {'logger_level': 30}
    for imex in [True, False]:
        custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-6, 'dt_max': dt_max[imex]}}
        stats, controller, _ = run_leaky_superconductor(
            custom_description=custom_description,
            custom_controller_params=custom_controller_params,
            imex=imex,
            Tend=5e2,
        )

        res[imex] = get_sorted(stats, type='u')[-1][1]
        if plotting:  # pragma no cover
            plot_solution(stats, controller)

    diff = abs(res[True] - res[False])
    assert (
        diff < 5e-3
    ), f"Difference between IMEX and fully-implicit too large! Got {diff:.2e}, allowed is only {5e-3:.2e}!"
    prob = controller.MS[0].levels[0].prob
    assert (
        max(res[True]) > prob.params.u_max
    ), f"Expected runaway to happen, but maximum temperature is {max(res[True]):.2e} < u_max={prob.params.u_max:.2e}!"


if __name__ == '__main__':
    compare_imex_full(plotting=True)
    plt.show()
