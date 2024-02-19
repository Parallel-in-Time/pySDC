# script to run a Brusselator problem
from pySDC.implementations.problem_classes.Brusselator import Brusselator
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.Errors import ConvergenceError


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
    step_params['maxiter'] = 9

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


if __name__ == '__main__':
    live_plot()
