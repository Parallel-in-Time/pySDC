import os

import dill
import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
from pySDC.projects.Hamiltonian.stop_at_error_hook import stop_at_error_hook


def run_simulation():
    """
    Routine to run the simulation of a second order problem

    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0.0
    level_params['dt'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['k'] = None  # will be defined later
    problem_params['phase'] = 0.0
    problem_params['amp'] = 1.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = stop_at_error_hook
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = harmonic_oscillator
    description['sweeper_class'] = verlet
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles

    # set time parameters
    t0 = 0.0
    Tend = 4.0
    num_procs = 4

    rlim_left = 0
    rlim_right = 16.0
    nstep = 34
    ks = np.linspace(rlim_left, rlim_right, nstep)[1:]

    # qd_combinations = [('IE', 'EE'), ('IE', 'PIC'),
    #                    ('LU', 'EE'), ('LU', 'PIC'),
    #                    # ('MIN3', 'PIC'), ('MIN3', 'EE'),
    #                    ('PIC', 'EE'), ('PIC', 'PIC')]
    qd_combinations = [('IE', 'EE'), ('PIC', 'PIC')]

    results = dict()
    results['ks'] = ks

    for qd in qd_combinations:
        print('Working on combination (%s, %s)...' % qd)

        niters = np.zeros(len(ks))

        for i, k in enumerate(ks):
            problem_params['k'] = k
            description['problem_params'] = problem_params

            sweeper_params['QI'] = qd[0]
            sweeper_params['QE'] = qd[1]
            description['sweeper_params'] = sweeper_params

            # instantiate the controller
            controller = controller_nonMPI(
                num_procs=num_procs, controller_params=controller_params, description=description
            )

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t=t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            uex = P.u_exact(Tend)

            print('Error after run: %s' % abs(uex - uend))

            # filter statistics by type (number of iterations)
            iter_counts = get_sorted(stats, type='niter', sortby='time')

            niters[i] = np.mean(np.array([item[1] for item in iter_counts]))

            # print('Worked on k = %s, took %s iterations' % (k, results[i]))

        results[qd] = niters

    fname = 'data/harmonic_k.dat'
    f = open(fname, 'wb')
    dill.dump(results, f)
    f.close()

    assert os.path.isfile(fname), 'Run did not create stats file'


def show_results(cwd=''):
    """
    Helper function to plot the error of the Hamiltonian

    Args:
        cwd (str): current working directory
    """

    plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=0.89)

    # read in the dill data
    f = open(cwd + 'data/harmonic_k.dat', 'rb')
    results = dill.load(f)
    f.close()

    ks = results['ks']

    for qd in results:
        if qd != 'ks':
            plt_helper.plt.plot(ks, results[qd], label=qd)

    plt_helper.plt.xlabel('k')
    plt_helper.plt.ylabel('Number of iterations')
    plt_helper.plt.legend(
        loc='upper left',
    )
    plt_helper.plt.ylim([0, 15])

    fname = 'data/harmonic_qd_iterations'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def main():
    run_simulation()
    show_results()


if __name__ == "__main__":
    main()
