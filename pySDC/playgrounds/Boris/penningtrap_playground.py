import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.datatype_classes.particles import particles, fields
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.playgrounds.Boris.penningtrap_HookClass import particles_output


def main():
    """
    Particle cloud in a penning trap, incl. live visualization

    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.015625

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 25.0
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]])
    problem_params['nparts'] = 10
    problem_params['sig'] = 0.1
    problem_params['Tend'] = 16.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    # description['space_transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 128 * 0.015625

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    extract_stats = filter_stats(stats, type='etot')
    sortedlist_stats = sort_stats(extract_stats, sortby='time')

    energy = [entry[1] for entry in sortedlist_stats]

    plt.figure()
    plt.plot(energy, 'bo--')

    plt.xlabel('Time')
    plt.ylabel('Energy')

    plt.savefig('penningtrap_energy.png', rasterized=True, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
