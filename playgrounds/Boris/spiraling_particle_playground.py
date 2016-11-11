from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.datatype_classes.particles import particles, fields

from pySDC.plugins.stats_helper import filter_stats, sort_stats
from playgrounds.Boris.spiraling_particle_ProblemClass import planewave_single
from playgrounds.Boris.spiraling_particle_HookClass import particles_output


def main(dt, Tend):

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['delta'] = 1
    problem_params['a0'] = 0.07
    problem_params['u0'] = np.array([[0, -1, 0], [0.05, 0.01, 0], [1], [1]])

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30
    # controller_params['log_to_file'] = True
    # controller_params['fname'] = 'step_3_B_out.txt'

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = planewave_single
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    # description['space_transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = Tend

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return uinit, stats, problem_params['a0']


def plot_error_and_positions(uinit, stats, a0):

    extract_stats = filter_stats(stats, type='energy')
    sortedlist_stats = sort_stats(extract_stats, sortby='time')

    R0 = np.linalg.norm(uinit.pos.values[:])
    H0 = 1 / 2 * np.dot(uinit.vel.values[:], uinit.vel.values[:]) + a0 / R0

    energy_err = [abs(entry[1] - H0) / H0 for entry in sortedlist_stats]

    plt.figure()
    plt.plot(energy_err, 'bo--')

    plt.xlabel('Time')
    plt.ylabel('Error in hamiltonian')

    plt.savefig('spiraling_particle_error_ham.png', rasterized=True, transparent=True, bbox_inches='tight')

    extract_stats = filter_stats(stats, type='position')
    sortedlist_stats = sort_stats(extract_stats, sortby='time')

    xpositions = [item[1][0] for item in sortedlist_stats]
    ypositions = [item[1][1] for item in sortedlist_stats]

    plt.figure()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(xpositions, ypositions)
    plt.savefig('spiraling_particle_positons.png', rasterized=True, transparent=True, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    uinit, stats, a0 = main(dt=1.0, Tend=5000)
    plot_error_and_positions(uinit, stats, a0)
