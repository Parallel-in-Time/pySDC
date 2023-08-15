# It checks whether data folder exicits or not
exec(open("check_data_folder.py").read())

import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import RKN
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import fixed_plot_params


def main(dt, tend, maxiter, M, sweeper):  # pragma: no cover
    """
    Implementation of Hamiltonian error for Harmonic oscillator problem
    mu=0
    kappa=1
    omega=1
    Args:
        dt: time step
        tend: final time
        maxiter: maximal iteration
        M: Number of quadrature nodes
        sweeper: sweeper class
    returns:
        Ham_err
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = dt
    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = M

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 1
    problem_params['omega_B'] = 0
    problem_params['u0'] = np.array([[0, 0, 0], [0, 0, 1], [1], [1]], dtype=object)
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1
    # problem_params['Tend'] = 16.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30
    penningtrap.Harmonic_oscillator = True
    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    # description['space_transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = tend

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    sortedlist_stats = get_sorted(stats, type='etot', sortby='time')

    # energy = [entry[1] for entry in sortedlist_stats]
    H0 = 1 / 2 * (np.dot(uinit.vel[:].T, uinit.vel[:]) + np.dot(uinit.pos[:].T, uinit.pos[:]))

    Ham_err = np.ravel([abs(entry[1] - H0) / H0 for entry in sortedlist_stats])
    return Ham_err


def plot_Hamiltonian_error(K, M, dt):  # pragma: no cover
    """
    Plot Hamiltonian Error
    Args:
        K: list of maxiter
        M: number of quadrature nodes
        dt: time step
    """
    fixed_plot_params()
    # Define final time
    time = 1e6
    tn = dt
    # Find time nodes
    t = np.arange(0, time + tn, tn)
    # Get saved data
    t_len = len(t)
    RKN1_ = np.loadtxt('data/Ham_RKN1.csv', delimiter='*')
    SDC2_ = np.loadtxt('data/Ham_SDC{}{}.csv'.format(M, K[0]), delimiter='*')
    SDC3_ = np.loadtxt('data/Ham_SDC{}{}.csv'.format(M, K[1]), delimiter='*')
    SDC4_ = np.loadtxt('data/Ham_SDC{}{}.csv'.format(M, K[2]), delimiter='*')
    # Only save Hamiltonian error
    RKN1 = RKN1_[:, 1]
    SDC2 = SDC2_[:, 1]
    SDC3 = SDC3_[:, 1]
    SDC4 = SDC4_[:, 1]
    step = 3000
    # plot Hamiltonian error
    plt.loglog(t[:t_len:step], RKN1[:t_len:step], label='RKN-4', marker='.', linestyle=' ')
    plt.loglog(t[:t_len:step], SDC2[:t_len:step], label='K={}'.format(K[0]), marker='s', linestyle=' ')
    plt.loglog(t[:t_len:step], SDC3[:t_len:step], label='K={}'.format(K[1]), marker='*', linestyle=' ')
    plt.loglog(t[:t_len:step], SDC4[:t_len:step], label='K={}'.format(K[2]), marker='H', linestyle=' ')
    plt.xlabel('$\omega \cdot t$')
    plt.ylabel('$\Delta H^{\mathrm{(rel)}}$')
    plt.ylim(1e-11, 1e-3 + 0.001)
    plt.legend(fontsize=15)
    plt.tight_layout()


if __name__ == "__main__":
    """
    Important:
        * Every simulation needs to run individually otherwise it may crash at some point.
            I don't know why
        * All of the data saved in /data folder
    """
    K = (2, 3, 4)
    M = 3
    dt = 2 * np.pi / 10
    tend = 2 * np.pi * 1e6

    Ham_SDC1 = main(dt, tend, K[0], M, boris_2nd_order)

    # Ham_SDC2=main(dt, tend, K[1], M, boris_2nd_order)

    # Ham_SDC3=main(dt, tend, K[2], M, boris_2nd_order)

    # Ham_RKN=main(dt, tend, 1, M, RKN)

    # plot_Hamiltonian_error(K, M, dt)
