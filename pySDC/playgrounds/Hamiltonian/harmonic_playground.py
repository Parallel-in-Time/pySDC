from __future__ import division
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import dill

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.playgrounds.Hamiltonian.hamiltonian_output import hamiltonian_output


def run_simulation():
    """
    Particle cloud in a penning trap, incl. live visualization

    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5]
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

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params,
                                             description=description)
    # controller = allinclusive_multigrid_nonMPI(num_procs=100, controller_params=controller_params,
    #                                            description=description)

    # set time parameters
    t0 = 0.0
    Tend = 5000.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=t0)
    uex = P.u_exact(t=Tend)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %2i' % item
        print(out)

    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % \
          (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    f = open('data/harmonic.dat', 'wb')
    dill.dump(stats, f)
    f.close()


def show_results():
    f = open('data/harmonic.dat', 'rb')
    stats = dill.load(f)
    f.close()
    # infile = np.load('data/harmonic.npz')
    # stats = infile['stats']
    extract_stats = filter_stats(stats, type='err_hamiltonian')
    result = defaultdict(list)
    for k, v in extract_stats.items():
        result[k.iter].append((k.time, v))
    for k, v in result.items():
        result[k] = sorted(result[k], key=lambda x: x[0])

    plt.figure()
    for k, v in result.items():
        time = [item[0] for item in v]
        ham = [item[1] for item in v]
        plt.semilogy(time, ham, '-', lw=1, label='Iter ' + str(k))
    #
    plt.xlabel('Time')
    plt.ylabel('Error in Hamiltonian')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    plt.savefig('harmonic_hamiltonian.png', rasterized=True, transparent=False, bbox_inches='tight')


if __name__ == "__main__":
    run_simulation()
    show_results()
