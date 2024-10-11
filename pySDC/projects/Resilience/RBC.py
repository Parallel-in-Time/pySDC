# script to run a Rayleigh-Benard Convection problem
from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE
from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.core.convergence_controller import ConvergenceController

from pySDC.core.errors import ConvergenceError


class ReachTendExactly(ConvergenceController):

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": +50,
            "Tend": None,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, step, **kwargs):
        L = step.levels[0]
        L.status.dt_new = min([L.params.dt, self.params.Tend - L.time - L.dt])


def run_RBC(
    custom_description=None,
    num_procs=1,
    Tend=14.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    u0=None,
    t0=10,
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
    level_params['dt'] = 5e-4
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {}

    step_params = {}
    step_params['maxiter'] = 5

    convergence_controllers = {}
    convergence_controllers[ReachTendExactly] = {'Tend': Tend}

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    imex_1st_order.compute_residual = compute_residual_DAE

    RayleighBenard._u_exact = RayleighBenard.u_exact
    RayleighBenard.u_exact = u_exact

    description = {}
    description['problem_class'] = RayleighBenard
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

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


def u_exact(self, t, u_init=None, t_init=None, recompute=False):
    import pickle
    import os

    path = f'data/stats/RBC-u_init-{t}.pickle'
    if os.path.exists(path) and not recompute and t_init is None:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    else:
        from pySDC.helpers.stats_helper import get_sorted
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        convergence_controllers = {
            Adaptivity: {'e_tol': 1e-8, 'dt_rel_min_slope': 0.25},
        }
        desc = {'convergence_controllers': convergence_controllers}

        u0 = RayleighBenard()._u_exact(0) if u_init is None else u_init
        t0 = 0 if t_init is None else t_init
        stats, _, _ = run_RBC(Tend=t, u0=u0, t0=t0, custom_description=desc)

        u = get_sorted(stats, type='u', recomputed=False)[-1]
        data = u[1]

        if t0 == 0:
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    return data


if __name__ == '__main__':
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep

    stats, _, _ = run_RBC(hook_class=LogLocalErrorPostStep)
