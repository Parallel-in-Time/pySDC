import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.implementations.hooks.log_solution import LogSolution
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()

    sweeper_params['quad_type'] = 'LOBATTO'
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
    controller_params['hook_class'] = [LogSolution]

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 20

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'piline.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


def plot_voltages(cwd='./'):
    f = open(cwd + 'piline.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    u_val = get_sorted(stats, type='u', sortby='time')
    t = np.array([me[0] for me in u_val])
    v1 = np.array([me[1][0] for me in u_val])
    v2 = np.array([me[1][1] for me in u_val])
    p3 = np.array([me[1][2] for me in u_val])

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(t, v1, label='v1')
    plt_helper.plt.plot(t, v2, label='v2')
    plt_helper.plt.plot(t, p3, label='p3')
    plt_helper.plt.legend()

    plt_helper.plt.show()


if __name__ == "__main__":
    main()
    plot_voltages()
