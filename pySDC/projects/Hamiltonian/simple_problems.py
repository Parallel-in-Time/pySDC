from __future__ import division
from collections import defaultdict
import os
import dill

import pySDC.helpers.plot_helper as plt_helper

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from pySDC.implementations.problem_classes.HenonHeiles import henon_heiles
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles

from pySDC.helpers.stats_helper import filter_stats
from pySDC.projects.Hamiltonian.hamiltonian_output import hamiltonian_output


def setup_harmonic():
    """
    Helper routine for setting up everything for the harmonic oscillator

    Returns:
        description (dict): description of the controller
        controller_params (dict): controller parameters
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['spread'] = False

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['k'] = 1.0
    problem_params['phase'] = 0.0
    problem_params['amp'] = 1.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = hamiltonian_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = harmonic_oscillator
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = acceleration
    description['sweeper_class'] = verlet
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles

    return description, controller_params


def setup_henonheiles():
    """
    Helper routine for setting up everything for the Henon Heiles problem

    Returns:
        description (dict): description of the controller
        controller_params (dict): controller parameters
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['spread'] = False

    # initialize problem parameters for the Penning trap
    problem_params = dict()

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = hamiltonian_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = henon_heiles
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = acceleration
    description['sweeper_class'] = verlet
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles

    return description, controller_params


def run_simulation(prob=None):
    """
    Routine to run the simulation of a second order problem

    Args:
        prob (str): name of the problem

    """

    if prob == 'harmonic':
        description, controller_params = setup_harmonic()
        # set time parameters
        t0 = 0.0
        Tend = 50.0
        num_procs = 100
    elif prob == 'henonheiles':
        description, controller_params = setup_henonheiles()
        # set time parameters
        t0 = 0.0
        Tend = 25.0
        num_procs = 100
    else:
        raise NotImplemented('Problem type not implemented, got %s' % prob)

    # instantiate the controller
    controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                             description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'data/' + prob + '.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    assert os.path.isfile(fname), 'Run for %s did not create stats file' % prob


def show_results(prob=None, cwd=''):
    """
    Helper function to plot the error of the Hamiltonian

    Args:
        prob (str): name of the problem
        cwd (str): current working directory
    """
    f = open(cwd + 'data/' + prob + '.dat', 'rb')
    stats = dill.load(f)
    f.close()

    extract_stats = filter_stats(stats, type='err_hamiltonian')
    result = defaultdict(list)
    for k, v in extract_stats.items():
        result[k.iter].append((k.time, v))
    for k, v in result.items():
        assert k <= 6, 'Number of iterations is too high for %s, got %s' % (prob, k)
        result[k] = sorted(result[k], key=lambda x: x[0])

    plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=0.89)

    err_ham = 1
    for k, v in result.items():
        time = [item[0] for item in v]
        ham = [item[1] for item in v]
        err_ham = ham[-1]
        plt_helper.plt.semilogy(time, ham, '-', lw=1, label='Iter ' + str(k))

    assert err_ham < 2.3E-08, 'Error in the Hamiltonian is too large for %s, got %s' % (prob, err_ham)

    plt_helper.plt.xlabel('Time')
    plt_helper.plt.ylabel('Error in Hamiltonian')
    plt_helper.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fname = 'data/' + prob + '_hamiltonian'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def main():
    prob = 'harmonic'
    run_simulation(prob)
    show_results(prob)
    prob = 'henonheiles'
    run_simulation(prob)
    show_results(prob)


if __name__ == "__main__":
    main()
