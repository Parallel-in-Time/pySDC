# script to run a Rayleigh-Benard Convection problem
from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE, get_extrapolated_error_DAE
from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)
from pySDC.projects.Resilience.reachTendExactly import ReachTendExactly

from pySDC.core.errors import ConvergenceError

import numpy as np

PROBLEM_PARAMS = {'Rayleigh': 3.2e5, 'nx': 256, 'nz': 128, 'max_cached_factorizations': 30}


def u_exact(self, t, u_init=None, t_init=None, recompute=False, _t0=None):
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
            Adaptivity: {'e_tol': 1e-8, 'dt_rel_min_slope': 0.25, 'dt_min': 1e-5},
        }
        desc = {'convergence_controllers': convergence_controllers}

        u0 = self._u_exact(0) if u_init is None else u_init
        t0 = 0 if t_init is None else t_init

        if t == t0:
            return u0
        else:
            u0 = u0 if _t0 is None else self.u_exact(_t0)
            _t0 = t0 if _t0 is None else _t0

            stats, _, _ = run_RBC(Tend=t, u0=u0, t0=_t0, custom_description=desc)

        u = get_sorted(stats, type='u', recomputed=False)[-1]
        data = u[1]

        if t0 == 0:
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    return data


if not hasattr(RayleighBenard, '_u_exact'):
    RayleighBenard._u_exact = RayleighBenard.u_exact
    RayleighBenard.u_exact = u_exact


def run_RBC(
    custom_description=None,
    num_procs=1,
    Tend=21.0,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    u0=None,
    t0=20.0,
    use_MPI=False,
    step_size_rounding=False,
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
    EstimateExtrapolationErrorNonMPI.get_extrapolated_error = get_extrapolated_error_DAE

    level_params = {}
    level_params['dt'] = 1e-3
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    from mpi4py import MPI

    problem_params = {'comm': MPI.COMM_SELF, **PROBLEM_PARAMS}

    step_params = {}
    step_params['maxiter'] = 5

    convergence_controllers = {}
    convergence_controllers[ReachTendExactly] = {'Tend': Tend}
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeRounding

    if step_size_rounding:
        convergence_controllers[StepSizeRounding] = {}

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    imex_1st_order_efficient.compute_residual = compute_residual_DAE

    description = {}
    description['problem_class'] = RayleighBenard
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


def generate_data_for_fault_stats(Tend):
    prob = RayleighBenard(**PROBLEM_PARAMS)
    _ts = np.linspace(0, Tend, Tend * 10 + 1, dtype=float)
    for i in range(len(_ts) - 1):
        print(f'Generating reference solution from {_ts[i]:.4e} to {_ts[i+1]:.4e}')
        prob.u_exact(_ts[i + 1], _t0=_ts[i], recompute=False)


def plot_order(t, dt, steps, num_nodes, e_tol=1e-9, restol=1e-9, ax=None, recompute=False):  # pragma: no cover
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogSDCIterations, LogWork
    from pySDC.implementations.convergence_controller_classes.crash import StopAtNan
    from pySDC.helpers.stats_helper import get_sorted
    import pickle
    import os

    sweeper_params = {'num_nodes': num_nodes}
    level_params = {'e_tol': e_tol, 'restol': restol}
    step_params = {'maxiter': 99}
    convergence_controllers = {StopAtNan: {}}

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

            desc = {
                'sweeper_params': sweeper_params,
                'level_params': level_params,
                'step_params': step_params,
                'convergence_controllers': convergence_controllers,
            }

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

        if len(e) > 0:
            errors += [e[-1][1]]
        else:
            errors += [np.nan]

    errors = np.array(errors)
    dts = np.array(dts)
    mask = np.isfinite(errors)
    max_error = np.nanmax(errors)

    errors = errors[mask]
    dts = dts[mask]
    ax.loglog(dts, errors, label=f'{num_nodes} nodes', marker='x')
    ax.loglog(
        dts, [max_error * (me / dts[0]) ** (2 * num_nodes - 1) for me in dts], ls='--', label=f'order {2*num_nodes-1}'
    )
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'global error')
    ax.legend(frameon=False)


def check_order(t=14, dt=1e-1, steps=6):
    prob = RayleighBenard(**PROBLEM_PARAMS)
    _ts = [0, t, t + dt]
    for i in range(len(_ts) - 1):
        prob.u_exact(_ts[i + 1], _t0=_ts[i])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    for num_nodes in [1, 2, 3, 4]:
        plot_order(t=t, dt=dt, steps=steps, num_nodes=num_nodes, ax=ax, restol=1e-9)
    ax.set_title(f't={t:.2f}, dt={dt:.2f}')
    plt.show()


def plot_step_size(t0=0, Tend=30, e_tol=1e-3, recompute=False):  # pragma: no cover
    import matplotlib.pyplot as plt
    import pickle
    import os
    from pySDC.helpers.stats_helper import get_sorted

    path = f'data/stats/RBC-u-{t0:.8f}-{Tend:.8f}-{e_tol:.2e}.pickle'
    if os.path.exists(path) and not recompute:
        with open(path, 'rb') as file:
            stats = pickle.load(file)
    else:
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        convergence_controllers = {
            Adaptivity: {'e_tol': e_tol, 'dt_rel_min_slope': 0.25, 'dt_min': 1e-5},
        }
        desc = {'convergence_controllers': convergence_controllers}

        stats, _, _ = run_RBC(Tend=Tend, t0=t0, custom_description=desc)

        with open(path, 'wb') as file:
            pickle.dump(stats, file)

    fig, ax = plt.subplots(1, 1)

    dt = get_sorted(stats, type='dt', recomputed=False)
    ax.plot([me[0] for me in dt], [me[1] for me in dt])
    ax.set_ylabel(r'$\Delta t$')
    ax.set_xlabel(r'$t$')
    ax.set_yscale('log')
    plt.show()


def plot_factorizations_over_time(t0=0, Tend=50, e_tol=1e-3, recompute=False, adaptivity_mode='dt'):  # pragma: no cover
    import matplotlib.pyplot as plt
    import pickle
    import os
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityPolynomialError
    from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
    from pySDC.projects.Resilience.paper_plots import savefig

    setup_mpl()

    fig, axs = plt.subplots(1, 2, figsize=figsize_by_journal('TUHH_thesis', 1.0, 0.4))

    if adaptivity_mode == 'dt':
        adaptivity = Adaptivity
    elif adaptivity_mode == 'dt_k':
        adaptivity = AdaptivityPolynomialError

    dt_controllers = {
        'basic': {
            adaptivity: {
                'e_tol': e_tol,
            }
        },
        'min slope': {adaptivity: {'e_tol': e_tol, 'beta': 0.5, 'dt_rel_min_slope': 2}},
        'fixed': {},
        # 'rounding': {adaptivity: {'e_tol': e_tol, 'beta': 0.5, 'dt_rel_min_slope': 2}, StepSizeRounding: {}},
    }

    for name, params in dt_controllers.items():
        if adaptivity_mode == 'dt':
            path = f'data/stats/RBC-u-{t0:.8f}-{Tend:.8f}-{e_tol:.2e}-{name}.pickle'
        elif adaptivity_mode == 'dt_k':
            path = f'data/stats/RBC-u-{t0:.8f}-{Tend:.8f}-{e_tol:.2e}-{name}-dtk.pickle'

        if os.path.exists(path) and not recompute:
            with open(path, 'rb') as file:
                stats = pickle.load(file)
        else:

            convergence_controllers = {
                **params,
            }
            desc = {'convergence_controllers': convergence_controllers}

            if name == 'fixed':
                if adaptivity_mode == 'dt':
                    desc['level_params'] = {'dt': 2e-2}
                elif adaptivity_mode == 'dt_k':
                    desc['level_params'] = {'dt': 2e-3}
            elif adaptivity_mode == 'dt_k':
                desc['level_params'] = {'restol': 1e-7}

            stats, _, _ = run_RBC(
                Tend=Tend, t0=t0, custom_description=desc, hook_class=LogWork, step_size_rounding=False
            )

            with open(path, 'wb') as file:
                pickle.dump(stats, file)

        factorizations = get_sorted(stats, type='work_factorizations')
        rhs_evals = get_sorted(stats, type='work_rhs')
        axs[0].plot([me[0] for me in factorizations], np.cumsum([me[1] for me in factorizations]), label=name)
        axs[1].plot([me[0] for me in rhs_evals], np.cumsum([me[1] for me in rhs_evals]), label=name)

    axs[0].set_ylabel(r'matrix factorizations')
    axs[1].set_ylabel(r'right hand side evaluations')
    axs[0].set_xlabel(r'$t$')
    axs[1].set_xlabel(r'$t$')
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend(frameon=False)
    savefig(fig, f'RBC_step_size_controller_{adaptivity_mode}')


if __name__ == '__main__':
    # plot_step_size(0, 30)
    generate_data_for_fault_stats(Tend=30)
    # plot_factorizations_over_time(e_tol=1e-3, adaptivity_mode='dt')
    # plot_factorizations_over_time(recompute=False, e_tol=1e-5, adaptivity_mode='dt_k')
    # check_order(t=20, dt=1., steps=7)
    # stats, _, _ = run_RBC()
