# script to run a Gray-Scott problem
from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence
from pySDC.projects.Resilience.reachTendExactly import ReachTendExactly

from pySDC.core.errors import ConvergenceError

import numpy as np


def u_exact(self, t, u_init=None, t_init=None, recompute=False, _t0=None):
    import pickle
    import os

    path = f'data/stats/GS-u_init-{t:.8f}.pickle'
    if os.path.exists(path) and not recompute and t_init is None:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    else:
        from pySDC.helpers.stats_helper import get_sorted
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        convergence_controllers = {
            Adaptivity: {'e_tol': 1e-8},
        }
        desc = {'convergence_controllers': convergence_controllers}

        u0 = self._u_exact(0) if u_init is None else u_init
        t0 = 0 if t_init is None else t_init

        if t == t0:
            return u0
        else:
            u0 = u0 if _t0 is None else self.u_exact(_t0)
            _t0 = t0 if _t0 is None else _t0

            stats, _, _ = run_GS(Tend=t, u0=u0, t0=_t0, custom_description=desc)

        u = get_sorted(stats, type='u', recomputed=False)[-1]
        data = u[1]

        if t0 == 0:
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    return data


if not hasattr(grayscott_imex_diffusion, '_u_exact'):
    grayscott_imex_diffusion._u_exact = grayscott_imex_diffusion.u_exact
    grayscott_imex_diffusion.u_exact = u_exact


def run_GS(
    custom_description=None,
    num_procs=1,
    Tend=1e2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    u0=None,
    t0=0,
    use_MPI=False,
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
        u0 (dtype_u): Initial value
        t0 (float): Starting time
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        bool: If the code crashed
    """

    level_params = {}
    level_params['dt'] = 1e0
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'MIN-SR-S'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    from mpi4py import MPI

    problem_params = {
        'comm': MPI.COMM_SELF,
        'num_blobs': -48,
        'L': 2,
        'nvars': (128,) * 2,
        'A': 0.062,
        'B': 0.1229,
        'Du': 2e-5,
        'Dv': 1e-5,
    }

    step_params = {}
    step_params['maxiter'] = 5

    convergence_controllers = {}
    convergence_controllers[ReachTendExactly] = {'Tend': Tend}

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    description = {}
    description['problem_class'] = grayscott_imex_diffusion
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_efficient
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

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

    t0 = 0.0 if t0 is None else t0
    uinit = P.u_exact(t0) if u0 is None else u0

    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        prepare_controller_for_faults(controller, fault_stuff)

    crash = False
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError as e:
        print(f'Warning: Premature termination!: {e}')
        stats = controller.return_stats()
        crash = True
    return stats, controller, crash


if __name__ == '__main__':
    stats, _, _ = run_GS()
