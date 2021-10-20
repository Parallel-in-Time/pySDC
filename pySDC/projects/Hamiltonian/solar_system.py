import os
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

import dill
import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.FullSolarSystem import full_solar_system
from pySDC.implementations.problem_classes.OuterSolarSystem import outer_solar_system
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
from pySDC.projects.Hamiltonian.hamiltonian_output import hamiltonian_output


def setup_outer_solar_system():
    """
    Helper routine for setting up everything for the outer solar system problem

    Returns:
        description (dict): description of the controller
        controller_params (dict): controller parameters
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 100.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['sun_only'] = [False, True]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = hamiltonian_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = outer_solar_system
    description['problem_params'] = problem_params
    description['sweeper_class'] = verlet
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles

    return description, controller_params


def setup_full_solar_system():
    """
    Helper routine for setting up everything for the full solar system problem

    Returns:
        description (dict): description of the controller
        controller_params (dict): controller parameters
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 10.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['sun_only'] = [False, True]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = hamiltonian_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = full_solar_system
    description['problem_params'] = problem_params
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

    if prob == 'outer_solar_system':
        description, controller_params = setup_outer_solar_system()
        # set time parameters
        t0 = 0.0
        Tend = 10000.0
        num_procs = 100
        maxmeaniter = 6.0
    elif prob == 'full_solar_system':
        description, controller_params = setup_full_solar_system()
        # set time parameters
        t0 = 0.0
        Tend = 1000.0
        num_procs = 100
        maxmeaniter = 19.0
    else:
        raise NotImplementedError('Problem type not implemented, got %s' % prob)

    f = open(prob + '_out.txt', 'w')
    out = 'Running ' + prob + ' problem with %s processors...' % num_procs
    f.write(out + '\n')
    print(out)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                   description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    # for item in iter_counts:
    #     out = 'Number of iterations for time %4.2f: %2i' % item
    #     f.write(out)
    #     print(out)

    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    f.write(out + '\n')
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % \
          (int(np.argmax(niters)), int(np.argmin(niters)))
    f.write(out + '\n')
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    f.write(out + '\n')
    print(out)
    f.close()

    assert np.mean(niters) <= maxmeaniter, 'Mean number of iterations is too high, got %s' % np.mean(niters)

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

    # read in the dill data
    f = open(cwd + 'data/' + prob + '.dat', 'rb')
    stats = dill.load(f)
    f.close()

    plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()

    # extract error in hamiltonian and prepare for plotting
    extract_stats = filter_stats(stats, type='err_hamiltonian')
    result = defaultdict(list)
    for k, v in extract_stats.items():
        result[k.iter].append((k.time, v))
    for k, _ in result.items():
        result[k] = sorted(result[k], key=lambda x: x[0])

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    # Rearrange data for easy plotting
    err_ham = 1
    for k, v in result.items():
        time = [item[0] for item in v]
        ham = [item[1] for item in v]
        err_ham = ham[-1]
        plt_helper.plt.semilogy(time, ham, '-', lw=1, label='Iter ' + str(k))
    assert err_ham < 2.4E-14, 'Error in the Hamiltonian is too large for %s, got %s' % (prob, err_ham)

    plt_helper.plt.xlabel('Time')
    plt_helper.plt.ylabel('Error in Hamiltonian')
    plt_helper.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fname = 'data/' + prob + '_hamiltonian'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'

    # extract positions and prepare for plotting
    extract_stats = filter_stats(stats, type='position')
    result = sort_stats(extract_stats, sortby='time')

    fig = plt_helper.plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rearrange data for easy plotting
    nparts = len(result[1][1][0])
    ndim = len(result[1][1])
    nsteps = len(result)
    pos = np.zeros((nparts, ndim, nsteps))

    for idx, item in enumerate(result):
        for n in range(nparts):
            for m in range(ndim):
                pos[n, m, idx] = item[1][m][n]

    for n in range(nparts):
        if ndim == 2:
            ax.plot(pos[n, 0, :], pos[n, 1, :])
        elif ndim == 3:
            ax.plot(pos[n, 0, :], pos[n, 1, :], pos[n, 2, :])
        else:
            raise NotImplementedError('Wrong number of dimensions for plotting, got %s' % ndim)

    fname = 'data/' + prob + '_positions'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def main():
    prob = 'outer_solar_system'
    run_simulation(prob)
    show_results(prob)
    prob = 'full_solar_system'
    run_simulation(prob)
    show_results(prob)


if __name__ == "__main__":
    main()
