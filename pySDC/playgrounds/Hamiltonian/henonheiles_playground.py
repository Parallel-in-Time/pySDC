from __future__ import division
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import dill

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from pySDC.implementations.problem_classes.HenonHeiles import henon_heiles
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles

from pySDC.helpers.stats_helper import filter_stats
from pySDC.playgrounds.Hamiltonian.hamiltonian_output import hamiltonian_output


def run_simulation():
    """
    Particle cloud in a penning trap, incl. live visualization

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
    sweeper_params['QI'] = ['IE', 'PIC']
    sweeper_params['QE'] = ['EE', 'PIC']

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

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=100, controller_params=controller_params,
                                             description=description)
    # controller = allinclusive_multigrid_nonMPI(num_procs=1000, controller_params=controller_params,
    #                                            description=description)

    # set time parameters
    t0 = 0.0
    Tend = 50.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    print(uend.pos.values, uend.vel.values)

    f = open('data/henonheiles_x.dat', 'wb')
    dill.dump(stats, f)
    f.close()


def show_results():
    f = open('data/henonheiles_x.dat', 'rb')
    stats = dill.load(f)
    f.close()

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
    # plt.ylim([1E-10, 1E-03])
    #
    plt.savefig('henonheiles_hamiltonian_x.png', rasterized=True, transparent=False, bbox_inches='tight')


if __name__ == "__main__":
    run_simulation()
    show_results()
