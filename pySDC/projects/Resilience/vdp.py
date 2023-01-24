# script to run a van der Pol problem
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.core.Errors import ProblemError, ConvergenceError
from pySDC.projects.Resilience.hook import LogData, hook_collection


def plot_step_sizes(stats, ax):
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

    e_em = np.array(get_sorted(stats, type='error_embedded_estimate', recomputed=False, sortby='time'))[:, 1]
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
    custom_problem_params=None,
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
        custom_problem_params (dict): Overwrite presets
        use_MPI (bool): Whether or not to use MPI

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
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol  # pass problem class
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
        args = {'time': 1.0, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except (ProblemError, ConvergenceError):
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

        if comm is None or use_MPI == False:
            data[type] = [me[1] for me in get_sorted(stats, type=type, recomputed=None, sortby='time')]
        else:
            data[type] = [me[1] for me in get_sorted(stats, type=type, recomputed=None, sortby='time', comm=comm)]
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

    custom_controller_params = {'logger_level': 30}

    data = [{}, {}]

    for i in range(2):
        if use_MPI[i] or rank == 0:
            stats, controller, Tend = run_vdp(
                custom_description=custom_description,
                num_procs=size,
                use_MPI=use_MPI[i],
                custom_controller_params=custom_controller_params,
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
    custom_controller_params = {'logger_level': 30, 'all_to_done': False}
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
    custom_description = {'convergence_controllers': {}, 'level_params': {'dt': 1.0e-2}}
    custom_controller_params = {'logger_level': 30}
    expect = {}
    params = {'e_tol': 1e-6}

    for limit_step_sizes in [False, True]:
        if limit_step_sizes:
            params['dt_max'] = expect['dt_max'] * 0.9
            params['dt_min'] = expect['dt_min'] * 1.1
        else:
            for k in ['dt_max', 'dt_min']:
                if k in params.keys():
                    params.pop(k)

        custom_description['convergence_controllers'][Adaptivity] = params
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            num_procs=size,
            use_MPI=comm is not None,
            custom_controller_params=custom_controller_params,
            Tend=5.0e0,
            comm=comm,
        )

        # plot the step sizes
        dt = get_sorted(stats, type='dt', recomputed=False, comm=comm)

        if not limit_step_sizes:
            expect['dt_max'] = max([me[1] for me in dt])
            expect['dt_min'] = min([me[1] for me in dt])
        else:
            dt_max = max([me[1] for me in dt])
            dt_min = min([me[1] for me in dt[size:-size]])  # The first and last step might fall below the limits
            assert (
                dt_max <= params['dt_max']
            ), f"Exceeded maximum allowed step size! Got {dt_max:.4e}, allowed {params['dt_max']:.4e}."
            assert (
                dt_min >= params['dt_min']
            ), f"Exceeded minimum allowed step size! Got {dt_min:.4e}, allowed {params['dt_min']:.4e}."

    if comm == None:
        print(f'Passed step size limiter test with {size} ranks in nonMPI implementation')
    else:
        if comm.rank == 0:
            print(f'Passed step size limiter test with {size} ranks in MPI implementation')


if __name__ == "__main__":
    try:
        from mpi4py import MPI

        MPI_ready = True
        comm = MPI.COMM_WORLD
        size = comm.size
    except ModuleNotFoundError:
        MPI_ready = False
        comm = None
        size = 1

    mpi_vs_nonMPI(MPI_ready, comm)
    check_step_size_limiter(size, comm)

    if size == 1:
        check_adaptivity_with_avoid_restarts(comm=None, size=1)
