import pytest


def run_Lorenz(efficient, skip_residual_computation, num_procs=1):
    from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.projects.Resilience.sweepers import generic_implicit_efficient, generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-1
    level_params['residual_type'] = 'last_rel'

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['skip_residual_computation'] = (
        ('IT_CHECK', 'IT_FINE', 'IT_COARSE', 'IT_DOWN', 'IT_UP') if skip_residual_computation else ()
    )

    problem_params = {
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogSolution, LogWork, LogGlobalErrorPostRun]
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = LorenzAttractor
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit_efficient if efficient else generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=1.0)

    return stats


def run_Schroedinger(efficient=False, num_procs=1, skip_residual_computation=False):
    from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import nonlinearschroedinger_imex
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from mpi4py import MPI

    space_comm = MPI.COMM_SELF
    rank = space_comm.Get_rank()

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1e-8
    level_params['dt'] = 2e-01
    level_params['nsweeps'] = 1
    level_params['residual_type'] = 'last_rel'

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['skip_residual_computation'] = (
        ('IT_FINE', 'IT_COARSE', 'IT_DOWN', 'IT_UP') if skip_residual_computation else ()
    )

    # initialize problem parameters
    problem_params = {}
    problem_params['nvars'] = (128, 128)
    problem_params['spectral'] = False
    problem_params['c'] = 1.0
    problem_params['comm'] = space_comm

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30 if rank == 0 else 99
    controller_params['hook_class'] = [LogSolution, LogWork, LogGlobalErrorPostRun]
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_params'] = problem_params
    description['problem_class'] = nonlinearschroedinger_imex
    description['sweeper_class'] = imex_1st_order_efficient if efficient else imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }

    comm = MPI.COMM_SELF
    controller = controller_MPI(**controller_args, comm=comm)
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=1.0)
    return stats


@pytest.mark.base
def test_generic_implicit_efficient(skip_residual_computation=False):
    stats_normal = run_Lorenz(efficient=False, skip_residual_computation=skip_residual_computation)
    stats_efficient = run_Lorenz(efficient=True, skip_residual_computation=skip_residual_computation)
    assert_sameness(stats_normal, stats_efficient, 'generic_implicit')
    assert_benefit(stats_normal, stats_efficient)


@pytest.mark.base
def test_residual_skipping():
    stats_normal = run_Lorenz(efficient=True, skip_residual_computation=False)
    stats_efficient = run_Lorenz(efficient=True, skip_residual_computation=True)
    assert_sameness(stats_normal, stats_efficient, 'generic_implicit', check_residual=False)


@pytest.mark.mpi4py
def test_residual_skipping_with_residual_tolerance():
    stats_normal = run_Schroedinger(efficient=True, skip_residual_computation=False)
    stats_efficient = run_Schroedinger(efficient=True, skip_residual_computation=True)
    assert_sameness(stats_normal, stats_efficient, 'imex_first_order', check_residual=False)


@pytest.mark.mpi4py
def test_imex_first_order_efficient():
    stats_normal = run_Schroedinger(efficient=False)
    stats_efficient = run_Schroedinger(efficient=True)
    assert_sameness(stats_normal, stats_efficient, 'imex_first_order')
    assert_benefit(stats_normal, stats_efficient)


def assert_sameness(stats_normal, stats_efficient, sweeper_name, check_residual=True):
    from pySDC.helpers.stats_helper import get_sorted, get_list_of_types
    import numpy as np

    for me in get_list_of_types(stats_normal):
        normal = [you[1] for you in get_sorted(stats_normal, type=me)]
        if 'timing' in me or all(you is None for you in normal) or (not check_residual and 'residual' in me):
            continue
        elif 'work_rhs' in me:
            efficient = [me[1] for me in get_sorted(stats_efficient, type=me)]
            assert all(
                normal[i] >= efficient[i] for i in range(len(efficient))
            ), f'Efficient sweeper performs more right hand side evaluations than regular implementations of {sweeper_name} sweeper!'
        else:
            comp = [you[1] for you in get_sorted(stats_efficient, type=me)]
            assert np.allclose(
                normal, comp
            ), f'Stats don\'t match in type \"{me}\" for efficient and regular implementations of {sweeper_name} sweeper!'


def assert_benefit(stats_normal, stats_efficient):
    from pySDC.helpers.stats_helper import get_sorted

    rhs_evals_normal = sum([me[1] for me in get_sorted(stats_normal, type='work_rhs')])
    rhs_evals_efficient = sum([me[1] for me in get_sorted(stats_efficient, type='work_rhs')])
    assert (
        rhs_evals_normal > rhs_evals_efficient
    ), f"More efficient sweeper did not perform fewer rhs evaluations! ({rhs_evals_efficient} vs {rhs_evals_normal})"
