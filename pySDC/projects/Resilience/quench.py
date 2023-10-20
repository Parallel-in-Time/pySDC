# script to run a quench problem
from pySDC.implementations.problem_classes.Quench import Quench, QuenchIMEX
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
import numpy as np

import matplotlib.pyplot as plt
from pySDC.core.Errors import ConvergenceError


class live_plot(hooks):  # pragma: no cover
    """
    This hook plots the solution and the non-linear part of the right hand side after every step. Keep in mind that using adaptivity will result in restarts, which is not marked in these plots. Prepare to see the temperature profile jumping back again after a restart.
    """

    def _plot_state(self, step, level_number):  # pragma: no cover
        """
        Plot the solution at all collocation nodes and the non-linear part of the right hand side

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        L = step.levels[level_number]
        for ax in self.axs:
            ax.cla()
        # [self.axs[0].plot(L.prob.xv, L.u[i], label=f"node {i}") for i in range(len(L.u))]
        self.axs[0].plot(L.prob.xv, L.u[-1])
        self.axs[0].axhline(L.prob.u_thresh, color='black')
        self.axs[1].plot(L.prob.xv, L.prob.eval_f_non_linear(L.u[-1], L.time))
        self.axs[0].set_ylim(0, 0.025)
        self.fig.suptitle(f"t={L.time:.2e}, k={step.status.iter}")
        plt.pause(1e-1)

    def pre_run(self, step, level_number):  # pragma: no cover
        """
        Setup a figure to plot into

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))

    def post_step(self, step, level_number):  # pragma: no cover
        """
        Call the plotting function after the step

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        self._plot_state(step, level_number)


def run_quench(
    custom_description=None,
    num_procs=1,
    Tend=6e2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    imex=False,
    u0=None,
    t0=None,
    use_MPI=False,
    **kwargs,
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
        imex (bool): Solve the problem IMEX or fully implicit
        u0 (dtype_u): Initial value
        t0 (float): Starting time
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """
    if custom_description is not None:
        problem_params = custom_description.get('problem_params', {})
        if 'imex' in problem_params.keys():
            imex = problem_params['imex']
            problem_params.pop('imex', None)

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 10.0

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'newton_tol': 1e-9,
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = QuenchIMEX if imex else Quench
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order if imex else generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    # set time parameters
    t0 = 0.0 if t0 is None else t0

    # instantiate controller
    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(**controller_args, comm=comm)
        P = controller.S.levels[0].prob
    else:
        controller = controller_nonMPI(**controller_args, num_procs=num_procs)
        P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0) if u0 is None else u0

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        prepare_controller_for_faults(controller, fault_stuff)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError:
        print('Warning: Premature termination!')
        stats = controller.return_stats()
    return stats, controller, Tend


def faults(seed=0):  # pragma: no cover
    import matplotlib.pyplot as plt
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    fig, ax = plt.subplots(1, 1)

    rng = np.random.RandomState(seed)
    fault_stuff = {'rng': rng, 'args': {}, 'rnd_args': {}}

    controller_params = {'logger_level': 30}
    description = {'level_params': {'dt': 1e1}, 'step_params': {'maxiter': 5}}
    stats, controller, _ = run_quench(custom_controller_params=controller_params, custom_description=description)
    plot_solution_faults(stats, controller, ax, plot_lines=True, label='ref')

    stats, controller, _ = run_quench(
        fault_stuff=fault_stuff,
        custom_controller_params=controller_params,
    )
    plot_solution_faults(stats, controller, ax, label='fixed')

    description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-7, 'dt_max': 1e2, 'dt_min': 1e-3}}
    stats, controller, _ = run_quench(
        fault_stuff=fault_stuff, custom_controller_params=controller_params, custom_description=description
    )

    plot_solution_faults(stats, controller, ax, label='adaptivity', ls='--')
    plt.show()


def plot_solution_faults(stats, controller, ax, plot_lines=False, **kwargs):  # pragma: no cover
    u_ax = ax

    u = get_sorted(stats, type='u', recomputed=False)
    u_ax.plot([me[0] for me in u], [np.mean(me[1]) for me in u], **kwargs)

    if plot_lines:
        P = controller.MS[0].levels[0].prob
        u_ax.axhline(P.u_thresh, color='grey', ls='-.', label=r'$T_\mathrm{thresh}$')
        u_ax.axhline(P.u_max, color='grey', ls=':', label=r'$T_\mathrm{max}$')

    [ax.axvline(me[0], color='grey', label=f'fault at t={me[0]:.2f}') for me in get_sorted(stats, type='bitflip')]

    u_ax.legend()
    u_ax.set_xlabel(r'$t$')
    u_ax.set_ylabel(r'$T$')


def get_crossing_time(stats, controller, num_points=5, inter_points=50, temperature_error_thresh=1e-5):
    """
    Compute the time when the temperature threshold is crossed based on interpolation.

    Args:
        stats (dict): The stats from a pySDC run
        controller (pySDC.Controller.controller): The controller
        num_points (int): The number of points in the solution you want to use for interpolation
        inter_points (int): The resolution of the interpolation
        temperature_error_thresh (float): The temperature error compared to the actual threshold you want to allow

    Returns:
        float: The time when the temperature threshold is crossed
    """
    from pySDC.core.Lagrange import LagrangeApproximation
    from pySDC.core.Collocation import CollBase

    P = controller.MS[0].levels[0].prob
    u_thresh = P.u_thresh

    u = get_sorted(stats, type='u', recomputed=False)
    temp = np.array([np.mean(me[1]) for me in u])
    t = np.array([me[0] for me in u])

    crossing_index = np.arange(len(temp))[temp > u_thresh][0]

    # interpolation stuff
    num_points = min([num_points, crossing_index * 2, len(temp) - crossing_index])
    idx = np.arange(num_points) - num_points // 2 + crossing_index
    t_grid = t[idx]
    u_grid = temp[idx]
    t_inter = np.linspace(t_grid[0], t_grid[-1], inter_points)
    interpolator = LagrangeApproximation(points=t_grid)
    u_inter = interpolator.getInterpolationMatrix(t_inter) @ u_grid

    crossing_inter = np.arange(len(u_inter))[u_inter > u_thresh][0]

    temperature_error = abs(u_inter[crossing_inter] - u_thresh)

    assert temperature_error < temp[crossing_index], "Temperature error is rising due to interpolation!"

    if temperature_error > temperature_error_thresh and inter_points < 300:
        return get_crossing_time(stats, controller, num_points + 4, inter_points + 15, temperature_error_thresh)

    return t_inter[crossing_inter]


def plot_solution(stats, controller):  # pragma: no cover
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    u_ax = ax
    dt_ax = u_ax.twinx()

    u = get_sorted(stats, type='u', recomputed=False)
    u_ax.plot([me[0] for me in u], [np.mean(me[1]) for me in u], label=r'$T$')

    dt = get_sorted(stats, type='dt', recomputed=False)
    dt_ax.plot([me[0] for me in dt], [me[1] for me in dt], color='black', ls='--')
    u_ax.plot([None], [None], color='black', ls='--', label=r'$\Delta t$')

    if controller.useMPI:
        P = controller.S.levels[0].prob
    else:
        P = controller.MS[0].levels[0].prob
    u_ax.axhline(P.u_thresh, color='grey', ls='-.', label=r'$T_\mathrm{thresh}$')
    u_ax.axhline(P.u_max, color='grey', ls=':', label=r'$T_\mathrm{max}$')

    [ax.axvline(me[0], color='grey', label=f'fault at t={me[0]:.2f}') for me in get_sorted(stats, type='bitflip')]

    u_ax.legend()
    u_ax.set_xlabel(r'$t$')
    u_ax.set_ylabel(r'$T$')
    dt_ax.set_ylabel(r'$\Delta t$')


def compare_imex_full(plotting=False, leak_type='linear'):
    """
    Compare the results of IMEX and fully implicit runs.

    Args:
        plotting (bool): Plot the solution or not
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

    maxiter = 5
    num_nodes = 3
    newton_maxiter = 99

    res = {}
    rhs = {}
    error = {}

    custom_description = {}
    custom_description['problem_params'] = {
        'newton_tol': 1e-10,
        'newton_maxiter': newton_maxiter,
        'nvars': 2**7,
        'leak_type': leak_type,
    }
    custom_description['step_params'] = {'maxiter': maxiter}
    custom_description['sweeper_params'] = {'num_nodes': num_nodes}
    custom_description['convergence_controllers'] = {
        Adaptivity: {'e_tol': 1e-7},
    }

    custom_controller_params = {'logger_level': 15}
    for imex in [False, True]:
        stats, controller, _ = run_quench(
            custom_description=custom_description,
            custom_controller_params=custom_controller_params,
            imex=imex,
            Tend=5e2,
            use_MPI=False,
            hook_class=[LogWork, LogGlobalErrorPostRun],
        )

        res[imex] = get_sorted(stats, type='u')[-1][1]
        newton_iter = [me[1] for me in get_sorted(stats, type='work_newton')]
        rhs[imex] = np.mean([me[1] for me in get_sorted(stats, type='work_rhs')]) // 1
        error[imex] = get_sorted(stats, type='e_global_post_run')[-1][1]

        if imex:
            assert all(me == 0 for me in newton_iter), "IMEX is not supposed to do Newton iterations!"
        else:
            assert max(newton_iter) / num_nodes / maxiter <= newton_maxiter, "Took more Newton iterations than allowed!"
        if plotting:  # pragma: no cover
            plot_solution(stats, controller)

    diff = abs(res[True] - res[False])
    thresh = 4e-3
    assert (
        diff < thresh
    ), f"Difference between IMEX and fully-implicit too large! Got {diff:.2e}, allowed is only {thresh:.2e}!"
    prob = controller.MS[0].levels[0].prob
    assert (
        max(res[True]) > prob.u_max
    ), f"Expected runaway to happen, but maximum temperature is {max(res[True]):.2e} < u_max={prob.u_max:.2e}!"

    assert (
        rhs[True] == rhs[False]
    ), f"Expected IMEX and fully implicit schemes to take the same number of right hand side evaluations per step, but got {rhs[True]} and {rhs[False]}!"

    assert error[True] < 1.2e-4, f'Expected error of IMEX version to be less than 1.2e-4, but got e={error[True]:.2e}!'
    assert (
        error[False] < 7.7e-5
    ), f'Expected error of fully implicit version to be less than 7.7e-5, but got e={error[False]:.2e}!'


def compare_reference_solutions_single():
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogLocalErrorPostStep
    from pySDC.implementations.hooks.log_solution import LogSolution

    types = ['DIRK', 'SDC', 'scipy']
    types = ['scipy']
    fig, ax = plt.subplots()
    error_ax = ax.twinx()
    Tend = 500

    colors = ['black', 'teal', 'magenta']

    from pySDC.projects.Resilience.strategies import AdaptivityStrategy, merge_descriptions, DoubleAdaptivityStrategy
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    strategy = DoubleAdaptivityStrategy()

    controller_params = {'logger_level': 15}

    for j in range(len(types)):
        description = {}
        description['level_params'] = {'dt': 5.0, 'restol': 1e-10}
        description['sweeper_params'] = {'QI': 'IE', 'num_nodes': 3}
        description['problem_params'] = {
            'leak_type': 'linear',
            'leak_transition': 'step',
            'nvars': 2**10,
            'reference_sol_type': types[j],
            'newton_tol': 1e-12,
        }

        description['level_params'] = {'dt': 5.0, 'restol': -1}
        description = merge_descriptions(description, strategy.get_custom_description(run_quench, 1))
        description['step_params'] = {'maxiter': 5}
        description['convergence_controllers'][Adaptivity]['e_tol'] = 1e-7

        stats, controller, _ = run_quench(
            custom_description=description,
            hook_class=[LogGlobalErrorPostStep, LogLocalErrorPostStep, LogSolution],
            Tend=Tend,
            imex=False,
            custom_controller_params=controller_params,
        )
        e_glob = get_sorted(stats, type='e_global_post_step', recomputed=False)
        e_loc = get_sorted(stats, type='e_local_post_step', recomputed=False)
        u = get_sorted(stats, type='u', recomputed=False)

        ax.plot([me[0] for me in u], [max(me[1]) for me in u], color=colors[j], label=f'{types[j]} reference')

        error_ax.plot([me[0] for me in e_glob], [me[1] for me in e_glob], color=colors[j], ls='--')
        error_ax.plot([me[0] for me in e_loc], [me[1] for me in e_loc], color=colors[j], ls=':')

    prob = controller.MS[0].levels[0].prob
    ax.axhline(prob.u_thresh, ls='-.', color='grey')
    ax.axhline(prob.u_max, ls='-.', color='grey')
    ax.plot([None], [None], ls='--', label=r'$e_\mathrm{global}$', color='grey')
    ax.plot([None], [None], ls=':', label=r'$e_\mathrm{local}$', color='grey')
    error_ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('solution')
    error_ax.set_ylabel('error')
    ax.set_title('Fully implicit quench problem')
    fig.tight_layout()
    fig.savefig('data/quench_refs_single.pdf', bbox_inches='tight')


def compare_reference_solutions():
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun, LogLocalErrorPostStep

    types = ['DIRK', 'SDC', 'scipy']
    fig, ax = plt.subplots()
    Tend = 500
    dt_list = [Tend / 2.0**me for me in [2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # dt_list = [Tend / 2.**me for me in [2, 3, 4, 5, 6, 7]]

    for j in range(len(types)):
        errors = [None] * len(dt_list)
        for i in range(len(dt_list)):
            description = {}
            description['level_params'] = {'dt': dt_list[i], 'restol': 1e-10}
            description['sweeper_params'] = {'QI': 'IE', 'num_nodes': 3}
            description['problem_params'] = {
                'leak_type': 'linear',
                'leak_transition': 'step',
                'nvars': 2**10,
                'reference_sol_type': types[j],
            }

            stats, controller, _ = run_quench(
                custom_description=description,
                hook_class=[LogGlobalErrorPostRun, LogLocalErrorPostStep],
                Tend=Tend,
                imex=False,
            )
            # errors[i] = get_sorted(stats, type='e_global_post_run')[-1][1]
            errors[i] = max([me[1] for me in get_sorted(stats, type='e_local_post_step', recomputed=False)])
            print(errors)
        ax.loglog(dt_list, errors, label=f'{types[j]} reference')

    ax.legend(frameon=False)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('global error')
    ax.set_title('Fully implicit quench problem')
    fig.tight_layout()
    fig.savefig('data/quench_refs.pdf', bbox_inches='tight')


def check_order(reference_sol_type='scipy'):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

    Tend = 500
    maxiter_list = [1, 2, 3, 4, 5]
    dt_list = [Tend / 2.0**me for me in [4, 5, 6, 7, 8, 9]]
    # dt_list = [Tend / 2.**me for me in [6, 7, 8]]

    fig, ax = plt.subplots()

    from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK43

    colors = ['black', 'teal', 'magenta', 'orange', 'red']
    for j in range(len(maxiter_list)):
        errors = [None] * len(dt_list)

        for i in range(len(dt_list)):
            description = {}
            description['level_params'] = {'dt': dt_list[i]}
            description['step_params'] = {'maxiter': maxiter_list[j]}
            description['sweeper_params'] = {'QI': 'IE', 'num_nodes': 3}
            description['problem_params'] = {
                'leak_type': 'linear',
                'leak_transition': 'step',
                'nvars': 2**10,
                'reference_sol_type': reference_sol_type,
            }
            description['convergence_controllers'] = {EstimateEmbeddedError: {}}

            # if maxiter_list[j] == 5:
            #    description['sweeper_class'] = DIRK43
            #    description['sweeper_params'] = {'maxiter': 1}

            stats, controller, _ = run_quench(
                custom_description=description, hook_class=[LogGlobalErrorPostRun], Tend=Tend, imex=False
            )
            # errors[i] = max([me[1] for me in get_sorted(stats, type='error_embedded_estimate')])
            errors[i] = get_sorted(stats, type='e_global_post_run')[-1][1]
            print(errors)
        ax.loglog(dt_list, errors, color=colors[j], label=f'{maxiter_list[j]} iterations')
        ax.loglog(
            dt_list, [errors[0] * (me / dt_list[0]) ** maxiter_list[j] for me in dt_list], color=colors[j], ls='--'
        )

    dt_list = np.array(dt_list)
    errors = np.array(errors)
    orders = np.log(errors[1:] / errors[:-1]) / np.log(dt_list[1:] / dt_list[:-1])
    print(orders, np.mean(orders))

    # ax.loglog(dt_list, local_errors)
    ax.legend(frameon=False)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('global error')
    # ax.set_ylabel('max. local error')
    ax.set_title('Fully implicit quench problem')
    fig.tight_layout()
    fig.savefig(f'data/order_quench_{reference_sol_type}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    compare_reference_solutions_single()
    # for reference_sol_type in ['DIRK', 'SDC', 'scipy']:
    #   check_order(reference_sol_type=reference_sol_type)
    # faults(19)
    # get_crossing_time()
    # compare_imex_full(plotting=True)
    plt.show()
