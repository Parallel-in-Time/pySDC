import numpy as np
import dill

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.playgrounds.Piline.log_data import log_data
import pySDC.helpers.plot_helper as plt_helper

def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 100.0
    problem_params['Rs'] = 1.0
    problem_params['C1'] = 1.0
    problem_params['Rpi'] = 0.2
    problem_params['C2'] = 1.0
    problem_params['Lpi'] = 1.0
    problem_params['Rl'] = 5.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline                  # pass problem class
    description['problem_params'] = problem_params                # pass problem parameters
    description['sweeper_class'] = imex_1st_order              # pass sweeper
    description['sweeper_params'] = sweeper_params                # pass sweeper parameters
    description['level_params'] = level_params                    # pass level parameters
    description['step_params'] = step_params                      # pass step parameters

    # set time parameters
    t0 = 0.0
    Tend = 20

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'data/piline.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


def plot_voltages(cwd='./'):
    f = open(cwd + 'data/piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = sort_stats(filter_stats(stats, type='v1'), sortby='time')
    v2 = sort_stats(filter_stats(stats, type='v2'), sortby='time')
    p3 = sort_stats(filter_stats(stats, type='p3'), sortby='time')

    times = [v[0] for v in v1]

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(times, [v[1] for v in v1], label='v1')
    plt_helper.plt.plot(times, [v[1] for v in v2], label='v2')
    plt_helper.plt.plot(times, [v[1] for v in p3], label='p3')
    plt_helper.plt.legend()

    plt_helper.plt.show()



if __name__ == "__main__":
    main()
    plot_voltages()
