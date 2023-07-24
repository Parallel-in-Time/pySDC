# script to run a van der Pol problem
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.core.Errors import ProblemError, ConvergenceError
from pySDC.projects.Resilience.hook import LogData, hook_collection
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import generic_implicit_efficient


def plot_step_sizes(stats, ax, e_em_key='error_embedded_estimate'):
    """
    Plot solution and step sizes to visualize the dynamics in the van der Pol equation.

    Args:
        stats (pySDC.stats): The stats object of the run
        ax: Somewhere to plot

    Returns:
        None
    """

    # convert filtered statistics to list of iterations count, sorted by process
    u = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    p = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
    t = np.array([me[0] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])

    e_em = np.array(get_sorted(stats, type=e_em_key, recomputed=False, sortby='time'))[:, 1]
    dt = np.array(get_sorted(stats, type='dt', recomputed=False, sortby='time'))
    restart = np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))

    ax.plot(t, u, label=r'$u$')
    ax.plot(t, p, label=r'$p$')

    dt_ax = ax.twinx()
    dt_ax.plot(dt[:, 0], dt[:, 1], color='black')
    dt_ax.plot(t, e_em, color='magenta')
    dt_ax.set_yscale('log')
    dt_ax.set_ylim((5e-10, 3e-1))

    ax.plot([None], [None], label=r'$\Delta t$', color='black')
    ax.plot([None], [None], label=r'$\epsilon_\mathrm{embedded}$', color='magenta')
    ax.plot([None], [None], label='restart', color='grey', ls='-.')

    for i in range(len(restart)):
        if restart[i, 1] > 0:
            ax.axvline(restart[i, 0], color='grey', ls='-.')
    ax.legend(frameon=False)

    ax.set_xlabel('time')


def plot_avoid_restarts(stats, ax, avoid_restarts):
    """
    Make a plot that shows how many iterations where required to solve to a point in time in the simulation.
    Also restarts are shown as vertical lines.

    Args:
        stats (pySDC.stats): The stats object of the run
        ax: Somewhere to plot
        avoid_restarts (bool): Whether the `avoid_restarts` option was set in order to choose a color

    Returns:
        None
    """
    sweeps = get_sorted(stats, type='sweeps', recomputed=None)
    restarts = get_sorted(stats, type='restart', recomputed=None)

    color = 'blue' if avoid_restarts else 'red'
    ls = ':' if not avoid_restarts else '-.'
    label = 'with' if avoid_restarts else 'without'

    ax.plot([me[0] for me in sweeps], np.cumsum([me[1] for me in sweeps]), color=color, label=f'{label} avoid_restarts')
    [ax.axvline(me[0], color=color, ls=ls) for me in restarts if me[1]]

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$k$')
    ax.legend(frameon=False)


def run_vdp(
    custom_description=None,
    num_procs=1,
    Tend=10.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    use_MPI=False,
    **kwargs,
):
    """
    Run a van der Pol problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit_efficient
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

        # get initial values on finest level
        P = controller.S.levels[0].prob
        uinit = P.u_exact(t0)
    else:
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        rnd_args = {'iteration': 3}
        # args = {'time': 0.9, 'target': 0}
        args = {'time': 5.25, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except (ProblemError, ConvergenceError):
        print('Warning: Premature termination!')
        stats = controller.return_stats()

    return stats, controller, Tend


def fetch_test_data(stats, comm=None, use_MPI=False):
    """
    Get data to perform tests on from stats

    Args:
        stats (pySDC.stats): The stats object of the run
        comm (mpi4py.MPI.Comm): MPI communicator, or `None` for the non-MPI version
        use_MPI (bool): Whether or not MPI was used when generating stats

    Returns:
        dict: Key values to perform tests on
    """
    types = ['error_embedded_estimate', 'restart', 'dt', 'sweeps', 'residual_post_step']
    data = {}
    for type in types:
        if type not in get_list_of_types(stats):
            raise ValueError(f"Can't read type \"{type}\" from stats, only got", get_list_of_types(stats))

        data[type] = [
            me[1] for me in get_sorted(stats, type=type, recomputed=None, sortby='time', comm=comm if use_MPI else None)
        ]

    # add time
    data['time'] = [
        me[0] for me in get_sorted(stats, type='u', recomputed=None, sortby='time', comm=comm if use_MPI else None)
    ]
    return data


def check_if_tests_match(data_nonMPI, data_MPI):
    """
    Check if the data matches between MPI and nonMPI versions

    Args:
        data_nonMPI (dict): Key values to perform tests on obtained without MPI
        data_MPI (dict): Key values to perform tests on obtained with MPI

    Returns:
        None
    """
    ops = [np.mean, np.min, np.max, len, sum]
    for type in data_nonMPI.keys():
        for op in ops:
            val_nonMPI = op(data_nonMPI[type])
            val_MPI = op(data_MPI[type])
            assert np.isclose(val_nonMPI, val_MPI), (
                f"Mismatch in operation {op.__name__} on type \"{type}\": with {data_MPI['size'][0]} ranks: "
                f"nonMPI: {val_nonMPI}, MPI: {val_MPI}"
            )
    print(f'Passed with {data_MPI["size"][0]} ranks')


def mpi_vs_nonMPI(MPI_ready, comm):
    """
    Check if MPI and non-MPI versions give the same output.

    Args:
        MPI_ready (bool): Whether or not we can use MPI at all
        comm (mpi4py.MPI.Comm): MPI communicator

    Returns:
        None
    """
    if MPI_ready:
        size = comm.size
        rank = comm.rank
        use_MPI = [True, False]
    else:
        size = 1
        rank = 0
        use_MPI = [False, False]

    if rank == 0:
        print(f"Running with {size} ranks")

    custom_description = {'convergence_controllers': {}}
    custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7, 'avoid_restarts': False}

    data = [{}, {}]

    for i in range(2):
        if use_MPI[i] or rank == 0:
            stats, controller, Tend = run_vdp(
                custom_description=custom_description,
                num_procs=size,
                use_MPI=use_MPI[i],
                Tend=1.0,
                comm=comm,
            )
            data[i] = fetch_test_data(stats, comm, use_MPI=use_MPI[i])
            data[i]['size'] = [size]

    if rank == 0:
        check_if_tests_match(data[1], data[0])


def check_adaptivity_with_avoid_restarts(comm=None, size=1):
    """
    Make a test if adaptivity with the option to avoid restarts based on a contraction factor estimate works as
    expected.
    To this end, we run the same test of the van der Pol equation twice with the only difference being this option
    turned off or on.
    We recorded how many iterations we expect to avoid by avoiding restarts and check against this value.
    Also makes a figure comparing the number of iterations over time.

    In principle there is an option to test MSSDC here, but this is only preliminary and needs to be checked further.

    Args:
       comm (mpi4py.MPI.Comm): MPI communicator, or `None` for the non-MPI version
       size (int): Number of steps for MSSDC, is overridden by communicator size if applicable

    Returns:
        None
    """
    fig, ax = plt.subplots()
    custom_description = {'convergence_controllers': {}, 'level_params': {'dt': 1.0e-2}}
    custom_controller_params = {'all_to_done': False}
    results = {'e': {}, 'sweeps': {}, 'restarts': {}}
    size = comm.size if comm is not None else size

    for avoid_restarts in [True, False]:
        custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7, 'avoid_restarts': avoid_restarts}
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            num_procs=size,
            use_MPI=comm is not None,
            custom_controller_params=custom_controller_params,
            Tend=10.0e0,
            comm=comm,
        )
        plot_avoid_restarts(stats, ax, avoid_restarts)

        # check error
        u = get_sorted(stats, type='u', recomputed=False)[-1]
        if comm is None:
            u_exact = controller.MS[0].levels[0].prob.u_exact(t=u[0])
        else:
            u_exact = controller.S.levels[0].prob.u_exact(t=u[0])
        results['e'][avoid_restarts] = abs(u[1] - u_exact)

        # check iteration counts
        results['sweeps'][avoid_restarts] = sum(
            [me[1] for me in get_sorted(stats, type='sweeps', recomputed=None, comm=comm)]
        )
        results['restarts'][avoid_restarts] = sum([me[1] for me in get_sorted(stats, type='restart', comm=comm)])

    fig.tight_layout()
    fig.savefig(f'data/vdp-{size}procs{"-use_MPI" if comm is not None else ""}-avoid_restarts.png')

    assert np.isclose(results['e'][True], results['e'][False], rtol=5.0), (
        'Errors don\'t match with avoid_restarts and without, got '
        f'{results["e"][True]:.2e} and {results["e"][False]:.2e}'
    )
    if size == 1:
        assert results['sweeps'][True] - results['sweeps'][False] == 1301 - 1344, (
            '{Expected to save 43 iterations '
            f"with avoid_restarts, got {results['sweeps'][False] - results['sweeps'][True]}"
        )
        assert results['restarts'][True] - results['restarts'][False] == 0 - 10, (
            '{Expected to save 10 restarts '
            f"with avoid_restarts, got {results['restarts'][False] - results['restarts'][True]}"
        )
        print('Passed avoid_restarts tests with 1 process')
    if size == 4:
        assert results['sweeps'][True] - results['sweeps'][False] == 2916 - 3008, (
            '{Expected to save 92 iterations '
            f"with avoid_restarts, got {results['sweeps'][False] - results['sweeps'][True]}"
        )
        assert results['restarts'][True] - results['restarts'][False] == 0 - 18, (
            '{Expected to save 18 restarts '
            f"with avoid_restarts, got {results['restarts'][False] - results['restarts'][True]}"
        )
        print('Passed avoid_restarts tests with 4 processes')


def check_step_size_limiter(size=4, comm=None):
    """
    Check the step size limiter convergence controller.
    First we run without step size limits and then enforce limits that are slightly above and below what the usual
    limits. Then we run again and see if we exceed the limits.

    Args:
        size (int): Number of steps for MSSDC
        comm (mpi4py.MPI.Comm): MPI communicator, or `None` for the non-MPI version

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

    custom_description = {'convergence_controllers': {}, 'level_params': {'dt': 1.0e-2}}
    expect = {}
    params = {'e_tol': 1e-6}

    for limit_step_sizes in [False, True]:
        if limit_step_sizes:
            params['dt_max'] = expect['dt_max'] * 0.9
            params['dt_min'] = np.inf
            params['dt_slope_max'] = expect['dt_slope_max'] * 0.9
            params['dt_slope_min'] = expect['dt_slope_min'] * 1.1
            custom_description['convergence_controllers'][StepSizeLimiter] = {'dt_min': expect['dt_min'] * 1.1}
        else:
            for k in ['dt_max', 'dt_min', 'dt_slope_max', 'dt_slope_min']:
                params.pop(k, None)
                custom_description['convergence_controllers'].pop(StepSizeLimiter, None)

        custom_description['convergence_controllers'][Adaptivity] = params
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            num_procs=size,
            use_MPI=comm is not None,
            Tend=5.0e0,
            comm=comm,
        )

        # plot the step sizes
        dt = get_sorted(stats, type='dt', recomputed=None, comm=comm)

        # make sure that the convergence controllers are only added once
        convergence_controller_classes = [type(me) for me in controller.convergence_controllers]
        for c in convergence_controller_classes:
            assert convergence_controller_classes.count(c) == 1, f'Convergence controller {c} added multiple times'

        dt_numpy = np.array([me[1] for me in dt])
        if not limit_step_sizes:
            expect['dt_max'] = max(dt_numpy)
            expect['dt_min'] = min(dt_numpy)
            expect['dt_slope_max'] = max(dt_numpy[:-2] / dt_numpy[1:-1])
            expect['dt_slope_min'] = min(dt_numpy[:-2] / dt_numpy[1:-1])
        else:
            dt_max = max(dt_numpy)
            dt_min = min(dt_numpy[size:-size])  # The first and last step might fall below the limits
            dt_slope_max = max(dt_numpy[:-2] / dt_numpy[1:-1])
            dt_slope_min = min(dt_numpy[:-2] / dt_numpy[1:-1])
            assert (
                dt_max <= expect['dt_max']
            ), f"Exceeded maximum allowed step size! Got {dt_max:.4e}, allowed {params['dt_max']:.4e}."
            assert (
                dt_min >= expect['dt_min']
            ), f"Exceeded minimum allowed step size! Got {dt_min:.4e}, allowed {params['dt_min']:.4e}."
            assert (
                dt_slope_max <= expect['dt_slope_max']
            ), f"Exceeded maximum allowed step size slope! Got {dt_slope_max:.4e}, allowed {params['dt_slope_max']:.4e}."
            assert (
                dt_slope_min >= expect['dt_slope_min']
            ), f"Exceeded minimum allowed step size slope! Got {dt_slope_min:.4e}, allowed {params['dt_slope_min']:.4e}."

            assert (
                dt_slope_max <= expect['dt_slope_max']
            ), f"Exceeded maximum allowed step size slope! Got {dt_slope_max:.4e}, allowed {params['dt_slope_max']:.4e}."
            assert (
                dt_slope_min >= expect['dt_slope_min']
            ), f"Exceeded minimum allowed step size slope! Got {dt_slope_min:.4e}, allowed {params['dt_slope_min']:.4e}."

    if comm is None:
        print(f'Passed step size limiter test with {size} ranks in nonMPI implementation')
    else:
        if comm.rank == 0:
            print(f'Passed step size limiter test with {size} ranks in MPI implementation')


def interpolation_stuff():  # pragma: no cover
    """
    Plot interpolation vdp with interpolation after a restart and compare it to other modes of adaptivity.
    """
    from pySDC.implementations.convergence_controller_classes.interpolate_between_restarts import (
        InterpolateBetweenRestarts,
    )
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.helpers.plot_helper import figsize_by_journal

    fig, axs = plt.subplots(4, 1, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1.0, 1.0), sharex=True)
    restart_ax = axs[2].twinx()

    colors = ['black', 'red', 'blue']
    labels = ['interpolate', 'regular', 'keep iterating']

    for i in range(3):
        convergence_controllers = {
            Adaptivity: {'e_tol': 1e-7, 'dt_max': 9.0e-1},
        }
        if i == 0:
            convergence_controllers[InterpolateBetweenRestarts] = {}
        if i == 2:
            convergence_controllers[Adaptivity]['avoid_restarts'] = True

        problem_params = {
            'mu': 5,
        }

        sweeper_params = {
            'QI': 'LU',
        }

        custom_description = {
            'convergence_controllers': convergence_controllers,
            'problem_params': problem_params,
            'sweeper_params': sweeper_params,
        }

        stats, controller, _ = run_vdp(
            custom_description=custom_description,
            hook_class=[LogLocalErrorPostStep, LogData, LogWork] + hook_collection,
        )

        k = get_sorted(stats, type='work_newton')
        restarts = get_sorted(stats, type='restart')
        u = get_sorted(stats, type='u', recomputed=False)
        e_loc = get_sorted(stats, type='e_local_post_step', recomputed=False)
        dt = get_sorted(stats, type='dt', recomputed=False)

        axs[0].plot([me[0] for me in u], [me[1][1] for me in u], color=colors[i], label=labels[i])
        axs[1].plot([me[0] for me in e_loc], [me[1] for me in e_loc], color=colors[i])
        axs[2].plot([me[0] for me in k], np.cumsum([me[1] for me in k]), color=colors[i])
        restart_ax.plot([me[0] for me in restarts], np.cumsum([me[1] for me in restarts]), color=colors[i], ls='--')
        axs[3].plot([me[0] for me in dt], [me[1] for me in dt], color=colors[i])

    for ax in [axs[1], axs[3]]:
        ax.set_yscale('log')
    axs[0].set_ylabel(r'$u$')
    axs[1].set_ylabel(r'$e_\mathrm{local}$')
    axs[2].set_ylabel(r'Newton iterations')
    restart_ax.set_ylabel(r'restarts (dashed)')
    axs[3].set_ylabel(r'$\Delta t$')
    axs[3].set_xlabel(r'$t$')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    try:
        from mpi4py import MPI

        MPI_ready = True
        comm = MPI.COMM_WORLD
        size = comm.size
    except ModuleNotFoundError:
        MPI_ready = False
        comm = None
        size = 1

    if len(sys.argv) == 1:
        mpi_vs_nonMPI(MPI_ready, comm)
        check_step_size_limiter(size, comm)

        if size == 1:
            check_adaptivity_with_avoid_restarts(comm=None, size=1)

    elif 'mpi_vs_nonMPI' in sys.argv:
        mpi_vs_nonMPI(MPI_ready, comm)
    elif 'check_step_size_limiter' in sys.argv:
        check_step_size_limiter(MPI_ready, comm)
    elif 'check_adaptivity_with_avoid_restarts' and size == 1:
        check_adaptivity_with_avoid_restarts(comm=None, size=1)
    else:
        raise NotImplementedError('Your test is not implemented!')
