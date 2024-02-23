# script to run a Brusselator problem
from pySDC.implementations.problem_classes.Brusselator import Brusselator
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pySDC.core.Errors import ConvergenceError


def add_reference_solution(problem_class, path):
    u_exact_regular = Brusselator.u_exact

    def u_exact_with_ref(prob, t, u_init=None, t_init=None):
        if t == 0:
            return u_exact_regular(prob, t)
        else:
            with open(path, 'rb') as file:
                data = pickle.load(file)
                u_init = prob.dtype_u(init=prob.init)
                u_init[...] = data[1][...]
            assert (
                t >= data[0]
            ), f'Requested exact solution at t={t:.2e}, but reference solution begins only at {data[0]:.2e}!'
            return u_exact_regular(prob, t, u_init=u_init, t_init=data[0])

    problem_class.u_exact = u_exact_with_ref


def run_Brusselator(
    custom_description=None,
    num_procs=1,
    Tend=10.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    imex=False,
    u0=None,
    t0=None,
    use_MPI=False,
    FFT=True,
    **kwargs,
):
    """
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
        bool: If the code crashed
    """

    level_params = {}
    level_params['dt'] = 1e-1
    level_params['restol'] = 1e-8

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'nvars': (128, 128),
    }

    step_params = {}
    step_params['maxiter'] = 99

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    description = {}
    description['problem_class'] = Brusselator
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_efficient
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    t0 = 0.0 if t0 is None else t0

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

    if fault_stuff is not None:
        raise NotImplementedError

    crash = False
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError as e:
        print(f'Warning: Premature termination!: {e}')
        stats = controller.return_stats()
        crash = True
    return stats, controller, crash


def live_plot():
    from pySDC.implementations.hooks.plotting import PlotPostStep
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

    PlotPostStep.plot_every = 1
    PlotPostStep.save_plot = 'data/movies/Brusselator'

    description = {}
    convergence_controllers = {
        # AdaptivityPolynomialError: {'e_tol': 1e-7, 'interpolate_between_restarts': False, 'dt_max': 1e+3},
    }
    description['convergence_controllers'] = convergence_controllers
    description['problem_params'] = {
        'alpha': 1e-1,
        'nvars': (128,) * 2,
    }

    controller_params = {'logger_level': 15}
    run_Brusselator(hook_class=PlotPostStep, custom_controller_params=controller_params, custom_description=description)


def compute_ref(store=False):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.helpers.stats_helper import get_sorted

    path = 'data/refs/trash.pickle'  # 'data/refs/Brusselator.pickle'
    add_reference_solution(Brusselator, path)

    description = {}
    description['level_params'] = {
        'dt': 4e-2,
        'restol': 1e-9,
    }
    controller_params = {'logger_level': 15}
    Tend = 1.001  # 9.86

    hooks = [LogSolution]
    if not store:
        hooks += [LogGlobalErrorPostRun]
    stats, _, _ = run_Brusselator(
        hook_class=hooks, custom_controller_params=controller_params, custom_description=description, Tend=Tend
    )

    u_final = get_sorted(stats, type='u')[-1]
    if store:
        with open(path, 'wb') as file:
            pickle.dump(u_final, file)
            print(f'Stored reference solution in {path!r}')


def check_order_ref(maxiters=None):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.helpers.stats_helper import get_sorted
    import tempfile

    tmp_file = tempfile.NamedTemporaryFile()

    path = tmp_file.name

    add_reference_solution(Brusselator, path)

    controller_params = {'logger_level': 30}
    Tend = 32e-2
    dt = Tend / 2**6

    # generate reference_solution
    description = {}
    description['level_params'] = {
        'dt': dt,
        'restol': 1e-12,
    }
    description['step_params'] = {
        'maxiter': 99,
    }
    description['sweeper_params'] = {
        'num_nodes': 5,
    }

    hooks = [LogSolution]
    stats, _, _ = run_Brusselator(
        hook_class=hooks, custom_controller_params=controller_params, custom_description=description, Tend=Tend - dt
    )

    u_ref = get_sorted(stats, type='u')[-1]
    with open(path, 'wb') as file:
        pickle.dump(u_ref, file)
        print(f'Stored reference solution in {path!r}')

    def compute_order(maxiter):
        dts = [Tend / me for me in [5, 8, 10, 16]]
        errors = []
        for dt in dts:
            description['level_params']['dt'] = dt
            description['level_params']['restol'] = -1
            description['step_params']['maxiter'] = maxiter
            description['sweeper_params']['num_nodes'] = 3
            stats, _, _ = run_Brusselator(
                hook_class=LogGlobalErrorPostRun,
                custom_controller_params=controller_params,
                custom_description=description,
                Tend=Tend,
            )
            errors += [get_sorted(stats, type='e_global_post_run')[-1][1]]

        errors = np.array(errors)
        dts = np.array(dts)
        orders = np.log(errors[1:] / errors[:-1]) / np.log(dts[1:] / dts[:-1])
        return np.median(orders)

    maxiters = [2, 3, 4, 5] if maxiters is None else maxiters
    for maxiter in maxiters:
        order = compute_order(maxiter)
        print(f'Expected order {maxiter}, got {order:.2f}')
        assert np.isclose(order, maxiter, atol=0.6), f'Expected order {maxiter}, got {order:.2f}'


if __name__ == '__main__':
    check_order_ref()
