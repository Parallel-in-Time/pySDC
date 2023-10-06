import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.EnergyGrids.log_data_battery import log_data_battery
import pySDC.helpers.plot_helper as plt_helper

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI


def main():
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = {
        'restol': -1,
        'dt': 2e-2,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'LOBATTO',
        'num_nodes': 5,
        'QI': 'LU',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 10,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
        'mssdc_jac': False,
    }

    # convergence controllers
    convergence_controllers = {}
    switch_estimator_params = {
        'tol': 1e-10,
        'alpha': 1.0,
    }
    convergence_controllers.update({SwitchEstimator: switch_estimator_params})
    adaptivity_params = {
        'e_tol': 1e-7,
    }
    convergence_controllers.update({Adaptivity: adaptivity_params})
    restarting_params = {
        'max_restarts': 50,
        'crash_after_max_restarts': False,
    }
    convergence_controllers.update({BasicRestartingNonMPI: restarting_params})

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': battery,
        'problem_params': {},  # use default problem params
        'sweeper_class': imex_1st_order,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    # set time parameters
    t0 = 0.0
    Tend = 0.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/piline.dat'
    fname = 'battery.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


def plot_voltages(cwd='./'):
    """
    Routine to plot the numerical solution of the model
    """

    f = open(cwd + 'battery.dat', 'rb')
    stats = dill.load(f)
    f.close()

    u = get_sorted(stats, type='u', sortby='time', recomputed=False)
    t = np.array([me[0] for me in u])
    iL = np.array([me[1][0] for me in u])
    vC = np.array([me[1][1] for me in u])

    dt = np.array(get_sorted(stats, type='dt', recomputed=False))
    list_gs = get_sorted(stats, type='restart')

    fig, ax = plt_helper.plt.subplots(1, 1)
    ax.plot(t, iL, label='$i_L$')
    ax.plot(t, vC, label='$v_C$')
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Energy', fontsize=20)
    for element in list_gs:
        if element[1] > 0:
            ax.axvline(element[0])
    dt_ax = ax.twinx()
    dt_ax.plot(dt[:, 0], dt[:, 1], 'ko--', label='dt')
    dt_ax.set_ylabel('dt', fontsize=20)
    dt_ax.set_yscale('log', base=10)

    ax.legend(frameon=False, loc='upper right')
    dt_ax.legend(frameon=False, loc='center right')

    fig.savefig('battery_adaptivity.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
    plot_voltages()
