# script to run a Rayleigh-Benard Convection problem
from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE, get_extrapolated_error_DAE
from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)

from pySDC.core.errors import ConvergenceError

import numpy as np


def u_exact(self, t, u_init=None, t_init=None, recompute=False):
    import pickle
    import os

    path = f'data/stats/RBC-u_init-{t:.8f}.pickle'
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

        u0 = self._u_exact(0) if u_init is None else u_init
        t0 = 0 if t_init is None else t_init

        if t == t0:
            return u0
        else:
            stats, _, _ = run_RBC(Tend=t, u0=u0, t0=t0, custom_description=desc)

        u = get_sorted(stats, type='u', recomputed=False)[-1]
        data = u[1]

        if t0 == 0:
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    return data


RayleighBenard._u_exact = RayleighBenard.u_exact
RayleighBenard.u_exact = u_exact
EstimateExtrapolationErrorNonMPI.get_extrapolated_error = get_extrapolated_error_DAE


class ReachTendExactly(ConvergenceController):

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": +50,
            "Tend": None,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, step, **kwargs):
        L = step.levels[0]
        dt = L.status.dt_new if L.status.dt_new else L.params.dt
        if self.params.Tend - L.time - L.dt < dt:
            L.status.dt_new = min([dt, self.params.Tend - L.time - L.dt])


def run_RBC(
    custom_description=None,
    num_procs=1,
    Tend=14.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    u0=None,
    t0=11,
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
    level_params['dt'] = 1e-3
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    from mpi4py import MPI

    problem_params = {'comm': MPI.COMM_SELF}

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

    description = {}
    description['problem_class'] = RayleighBenard
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
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


def generate_data_for_fault_stats():
    prob = RayleighBenard()
    for t in [11.0, 14.0]:
        prob.u_exact(t)


def plot_order(t, dt, steps, num_nodes, e_tol=1e-9, restol=1e-9, ax=None, recompute=False):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogSDCIterations, LogWork
    from pySDC.helpers.stats_helper import get_sorted
    import pickle
    import os

    sweeper_params = {'num_nodes': num_nodes}
    level_params = {'e_tol': e_tol, 'restol': restol}
    step_params = {'maxiter': 99}

    errors = []
    dts = []
    for i in range(steps):
        dts += [dt / 2**i]

        path = f'data/stats/RBC-u_order-{t:.8f}-{dts[-1]:.8e}-{num_nodes}-{e_tol:.2e}-{restol:.2e}.pickle'

        if os.path.exists(path) and not recompute:
            with open(path, 'rb') as file:
                stats = pickle.load(file)
        else:

            level_params['dt'] = dts[-1]

            desc = {'sweeper_params': sweeper_params, 'level_params': level_params, 'step_params': step_params}

            stats, _, _ = run_RBC(
                Tend=t + dt,
                t0=t,
                custom_description=desc,
                hook_class=[LogGlobalErrorPostRun, LogSDCIterations, LogWork],
            )

            with open(path, 'wb') as file:
                pickle.dump(stats, file)

        e = get_sorted(stats, type='e_global_post_run')
        # k = get_sorted(stats, type='k')

        errors += [e[-1][1]]

    errors = np.array(errors)
    dts = np.array(dts)
    mask = np.isfinite(errors)
    max_error = np.nanmax(errors)

    errors = errors[mask]
    dts = dts[mask]
    ax.loglog(dts, errors, label=f'{num_nodes} nodes')
    ax.loglog(
        dts, [max_error * (me / dts[0]) ** (2 * num_nodes - 1) for me in dts], ls='--', label=f'order {2*num_nodes-1}'
    )
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'global error')
    ax.legend(frameon=False)


def test_order(t=14, dt=1e-1, steps=6):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    for num_nodes in [1, 2, 3, 4]:
        plot_order(t=t, dt=dt, steps=steps, num_nodes=num_nodes, ax=ax)
    plt.show()


if __name__ == '__main__':
    generate_data_for_fault_stats()
    test_order()
    # stats, _, _ = run_RBC()
