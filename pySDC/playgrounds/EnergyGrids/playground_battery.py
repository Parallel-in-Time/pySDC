import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted

# from pySDC.helpers.visualization_tools import show_residual_across_simulation
from pySDC.implementations.collocations import Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.playgrounds.EnergyGrids.log_data_battery import log_data_battery
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1e-10
    level_params['e_tol'] = 1e-5
    level_params['dt'] = 2e-2

    # initialize sweeper parameters
    sweeper_params = dict()

    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 10
    problem_params['V_ref'] = 1

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['use_adaptivity'] = False
    controller_params['use_switch_estimator'] = True
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data_battery

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 2.4

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

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', sortby='time')
    vC = get_sorted(stats, type='voltage C', sortby='time')

    times = [v[0] for v in cL]

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(times, [v[1] for v in cL], label='current L')
    plt_helper.plt.plot(times, [v[1] for v in vC], label='voltage C')
    plt_helper.plt.legend()

    plt_helper.plt.show()


def plot_residuals(cwd='./'):
    """
    Routine to plot the residuals for one block of each iteration and each process
    """

    f = open(cwd + 'battery.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0
    f = open('battery_out.txt', 'w')
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])
    f.close()

    # call helper routine to produce residual plot

    fname = 'battery_residuals.png'
    show_residual_across_simulation(stats=stats, fname=fname)


if __name__ == "__main__":
    main()
    plot_voltages()
    # plot_residuals()
