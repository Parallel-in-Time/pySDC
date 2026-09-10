import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types

from pySDC.playgrounds.time_dep_BCs.heat_eq_time_dep_BCs import Heat1DTimeDependentBCs
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.hooks.plotting import PlotPostStep


def run_heat(dt=1e-1, Tend=4, kmax=5, ft=np.pi, plotting=False):
    level_params = {}
    level_params['dt'] = dt

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {'ft': ft, 'spectral_space': False}

    step_params = {}
    step_params['maxiter'] = kmax

    controller_params = {}
    controller_params['logger_level'] = 30
    if plotting:
        controller_params['hook_class'] = PlotPostStep

    description = {}
    description['problem_class'] = Heat1DTimeDependentBCs
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    t0 = 0.0

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return uend, stats, controller


def compute_errors(Tend, N, ft, kmax):
    solutions = []
    dts = []

    for n in range(N):
        dt = Tend / (2**n)
        u, _, _ = run_heat(dt=dt, Tend=Tend, ft=ft, kmax=kmax, plotting=False)

        solutions.append(u)
        dts.append(dt)

    errors = [abs(solutions[i] - solutions[-1]) / abs(solutions[-1]) for i in range(N - 1)]
    _dts = [dts[i] for i in range(N - 1)]

    return _dts, errors


def compute_order(dts, errors):
    orders = []
    for i in range(len(errors) - 1):
        if errors[i + 1] < 1e-12:
            break
        orders.append(np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1]))
    return np.median(orders)


def plot_order():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))

    for k in range(1, 10):
        dts, errors = compute_errors(4, 8, 0, k)
        order = compute_order(dts, errors)
        axs[0].loglog(dts, errors, label=f'{k=}, {order=:.1f}')

        dts, errors = compute_errors(4, 8, 1, k)
        order = compute_order(dts, errors)
        axs[1].loglog(dts, errors, label=f'{k=}, {order=:.1f}')

    axs[0].set_ylabel(r'relative global error')
    axs[0].set_xlabel(r'$\Delta t$')
    axs[1].set_xlabel(r'$\Delta t$')
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[0].set_title('Constant in time BCs')
    axs[1].set_title('Time-dependent BCs')
    fig.tight_layout()


if __name__ == '__main__':
    plot_order()
    plt.show()
