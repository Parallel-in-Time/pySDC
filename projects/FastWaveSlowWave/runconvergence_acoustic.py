import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
from matplotlib.ticker import ScalarFormatter

from projects.FastWaveSlowWave.HookClass_acoustic import dump_energy
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.problem_classes.AcousticAdvection_1D_FD_imex import acoustic_1d_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI


def compute_convergence_data():
    """
    Routine to run the 1d acoustic-advection example with different orders
    """

    num_procs = 1

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-14

    # This comes as read-in for the step class
    step_params = dict()

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['cadv'] = 0.1
    problem_params['cs'] = 1.00
    problem_params['order_adv'] = 5
    problem_params['waveno'] = 5

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['do_coll_update'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = acoustic_1d_imex
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['hook_class'] = dump_energy

    nsteps = np.zeros((3, 9))
    nsteps[0, :] = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    nsteps[1, :] = nsteps[0, :]
    nsteps[2, :] = nsteps[0, :]

    for order in [3, 4, 5]:

        error = np.zeros(np.shape(nsteps)[1])

        # setup parameters "in time"
        t0 = 0
        Tend = 1.0

        if order == 3:
            file = open('conv-data.txt', 'w')
        else:
            file = open('conv-data.txt', 'a')

        step_params['maxiter'] = order
        description['step_params'] = step_params

        for ii in range(0, np.shape(nsteps)[1]):

            ns = nsteps[order - 3, ii]
            if (order == 3) or (order == 4):
                problem_params['nvars'] = [(2, int(2 * ns))]
            elif order == 5:
                problem_params['nvars'] = [(2, int(2 * ns))]

            description['problem_params'] = problem_params

            dt = Tend / float(ns)

            level_params['dt'] = dt
            description['level_params'] = level_params

            # instantiate the controller
            controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                                     description=description)
            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)
            if ii == 0:
                print("Time step: %4.2f" % dt)
                print("Fast CFL number: %4.2f" % (problem_params['cs'] * dt / P.dx))
                print("Slow CFL number: %4.2f" % (problem_params['cadv'] * dt / P.dx))

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # compute exact solution and compare
            uex = P.u_exact(Tend)

            error[ii] = np.linalg.norm(uex.values - uend.values, np.inf) / np.linalg.norm(uex.values, np.inf)
            file.write(str(order) + "    " + str(ns) + "    " + str(error[ii]) + "\n")

        file.close()

        for ii in range(0, np.shape(nsteps)[1]):
            print('error for nsteps= %s: %s' % (nsteps[order - 3, ii], error[ii]))


def plot_convergence(cwd=''):
    """
    Plotting routine for the convergence data

    Args:
        cwd (string): current workign directory
    """

    fs = 8
    order = np.array([])
    nsteps = np.array([])
    error = np.array([])

    file = open(cwd + 'conv-data.txt', 'r')
    while True:
        line = file.readline()
        if not line:
            break
        items = str.split(line, "    ", 3)
        order = np.append(order, int(items[0]))
        nsteps = np.append(nsteps, int(float(items[1])))
        error = np.append(error, float(items[2]))

    assert np.size(order) == np.size(nsteps), 'Found different number of entries in order and nsteps'
    assert np.size(nsteps) == np.size(error), 'Found different number of entries in nsteps and error'

    assert np.size(nsteps) % 3 == 0, 'Number of entries not a multiple of three, got %s' % np.size(nsteps)

    N = int(np.size(nsteps) / 3)

    error_plot = np.zeros((3, N))
    nsteps_plot = np.zeros((3, N))
    convline = np.zeros((3, N))
    order_plot = np.zeros(3)

    for ii in range(0, 3):
        order_plot[ii] = order[N * ii]
        for jj in range(0, N):
            error_plot[ii, jj] = error[N * ii + jj]
            nsteps_plot[ii, jj] = nsteps[N * ii + jj]
            convline[ii, jj] = error_plot[ii, 0] * (float(nsteps_plot[ii, 0]) / float(nsteps_plot[ii, jj])) ** \
                order_plot[ii]

    color = ['r', 'b', 'g']
    shape = ['o', 'd', 's']
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    for ii in range(0, 3):
        plt.loglog(nsteps_plot[ii, :], convline[ii, :], '-', color=color[ii])
        plt.loglog(nsteps_plot[ii, :], error_plot[ii, :], shape[ii], markersize=fs, color=color[ii],
                   label='p=' + str(int(order_plot[ii])))

    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs})
    plt.xlabel('Number of time steps', fontsize=fs)
    plt.ylabel('Relative error', fontsize=fs, labelpad=2)
    plt.xlim([0.9 * np.min(nsteps_plot), 1.1 * np.max(nsteps_plot)])
    plt.ylim([1e-5, 1e0])
    plt.yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], fontsize=fs)
    plt.xticks([20, 30, 40, 60, 80, 100], fontsize=fs)
    plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    filename = 'convergence.png'
    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    compute_convergence_data()
    plot_convergence()
