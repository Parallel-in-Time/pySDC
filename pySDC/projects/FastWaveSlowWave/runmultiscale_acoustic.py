import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

from pySDC.projects.FastWaveSlowWave.HookClass_acoustic import dump_energy
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.acoustic_helpers.standard_integrators import bdf2, dirk, trapezoidal, rk_imex

from pySDC.projects.FastWaveSlowWave.AcousticAdvection_1D_FD_imex_multiscale import acoustic_1d_imex_multiscale


def compute_and_plot_solutions():
    """
    Routine to compute and plot the solutions of SDC(2), IMEX, BDF-2 and RK for a multiscale problem
    """

    num_procs = 1

    t0 = 0.0
    Tend = 3.0
    nsteps = 154  # 154 is value in Vater et al.
    dt = Tend / float(nsteps)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] = 2

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['cadv'] = 0.05
    problem_params['cs'] = 1.0
    problem_params['nvars'] = [(2, 512)]
    problem_params['order_adv'] = 5
    problem_params['waveno'] = 5

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 2

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = dump_energy

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = acoustic_1d_imex_multiscale
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params
    description['level_params'] = level_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # instantiate standard integrators to be run for comparison
    trap = trapezoidal((P.A + P.Dx).astype('complex'), 0.5)
    bdf2_m = bdf2(P.A + P.Dx)
    dirk_m = dirk((P.A + P.Dx).astype('complex'), step_params['maxiter'])
    rkimex = rk_imex(P.A.astype('complex'), P.Dx.astype('complex'), step_params['maxiter'])

    y0_tp = np.concatenate((uinit[0, :], uinit[1, :]))
    y0_bdf = y0_tp
    y0_dirk = y0_tp.astype('complex')
    y0_imex = y0_tp.astype('complex')

    # Perform time steps with standard integrators
    for i in range(0, nsteps):

        # trapezoidal rule step
        ynew_tp = trap.timestep(y0_tp, dt)

        # BDF-2 scheme
        if i == 0:
            ynew_bdf = bdf2_m.firsttimestep(y0_bdf, dt)
            ym1_bdf = y0_bdf
        else:
            ynew_bdf = bdf2_m.timestep(y0_bdf, ym1_bdf, dt)

        # DIRK scheme
        ynew_dirk = dirk_m.timestep(y0_dirk, dt)

        # IMEX scheme
        ynew_imex = rkimex.timestep(y0_imex, dt)

        y0_tp = ynew_tp
        ym1_bdf = y0_bdf
        y0_bdf = ynew_bdf
        y0_dirk = ynew_dirk
        y0_imex = ynew_imex

        # Finished running standard integrators
        unew_tp, pnew_tp = np.split(ynew_tp, 2)
        unew_bdf, pnew_bdf = np.split(ynew_bdf, 2)
        unew_dirk, pnew_dirk = np.split(ynew_dirk, 2)
        unew_imex, pnew_imex = np.split(ynew_imex, 2)

    fs = 8

    rcParams['figure.figsize'] = 2.5, 2.5
    # rcParams['pgf.rcfonts'] = False
    fig = plt.figure()

    sigma_0 = 0.1
    # k = 7.0 * 2.0 * np.pi
    x_0 = 0.75
    # x_1 = 0.25

    print('Maximum pressure in SDC: %5.3e' % np.linalg.norm(uend[1, :], np.inf))
    print('Maximum pressure in DIRK: %5.3e' % np.linalg.norm(pnew_dirk, np.inf))
    print('Maximum pressure in RK-IMEX: %5.3e' % np.linalg.norm(pnew_imex, np.inf))

    if dirk_m.order == 2:
        plt.plot(P.mesh, pnew_bdf, 'd-', color='c', label='BDF-2', markevery=(50, 75))
    p_slow = np.exp(-np.square(np.mod(P.mesh - problem_params['cadv'] * Tend, 1.0) - x_0) / (sigma_0 * sigma_0))
    plt.plot(P.mesh, p_slow, '--', color='k', markersize=fs - 2, label='Slow mode', dashes=(10, 2))
    if np.linalg.norm(pnew_imex, np.inf) <= 2:
        plt.plot(
            P.mesh, pnew_imex, '+-', color='r', label='IMEX(' + str(rkimex.order) + ')', markevery=(1, 75), mew=1.0
        )
    plt.plot(P.mesh, uend[1, :], 'o-', color='b', label='SDC(' + str(step_params['maxiter']) + ')', markevery=(25, 75))
    plt.plot(P.mesh, pnew_dirk, '-', color='g', label='DIRK(' + str(dirk_m.order) + ')')

    plt.xlabel('x', fontsize=fs, labelpad=0)
    plt.ylabel('Pressure', fontsize=fs, labelpad=0)
    fig.gca().set_xlim([0, 1.0])
    fig.gca().set_ylim([-0.5, 1.1])
    fig.gca().tick_params(axis='both', labelsize=fs)
    plt.legend(loc='upper left', fontsize=fs, prop={'size': fs}, handlelength=3)
    fig.gca().grid()
    filename = 'data/multiscale-K' + str(step_params['maxiter']) + '-M' + str(sweeper_params['num_nodes']) + '.png'
    plt.gcf().savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    compute_and_plot_solutions()
