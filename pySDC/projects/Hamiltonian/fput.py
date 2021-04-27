import os
from collections import defaultdict

import dill
import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.FermiPastaUlamTsingou import fermi_pasta_ulam_tsingou
from pySDC.implementations.sweeper_classes.verlet import verlet
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
from pySDC.projects.Hamiltonian.hamiltonian_and_energy_output import hamiltonian_and_energy_output


def setup_fput():
    """
    Helper routine for setting up everything for the Fermi-Pasta-Ulam-Tsingou problem

    Returns:
        description (dict): description of the controller
        controller_params (dict): controller parameters
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12
    level_params['dt'] = 2.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5, 3]
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['npart'] = 2048
    problem_params['alpha'] = 0.25
    problem_params['k'] = 1.0
    problem_params['energy_modes'] = [[1, 2, 3, 4]]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = hamiltonian_and_energy_output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = fermi_pasta_ulam_tsingou
    description['problem_params'] = problem_params
    description['sweeper_class'] = verlet
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles

    return description, controller_params


def run_simulation():
    """
    Routine to run the simulation of a second order problem

    """

    description, controller_params = setup_fput()
    # set time parameters
    t0 = 0.0
    # set this to 10000 to reproduce the picture in
    # http://www.scholarpedia.org/article/Fermi-Pasta-Ulam_nonlinear_lattice_oscillations
    Tend = 1000.0
    num_procs = 1

    f = open('fput_out.txt', 'w')
    out = 'Running fput problem with %s processors...' % num_procs
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
    #     f.write(out + '\n')
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

    # get runtime
    timing_run = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]
    out = '... took %6.4f seconds to run this.' % timing_run
    f.write(out + '\n')
    print(out)
    f.close()

    # assert np.mean(niters) <= 3.46, 'Mean number of iterations is too high, got %s' % np.mean(niters)

    fname = 'data/fput.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    assert os.path.isfile(fname), 'Run for %s did not create stats file'


def show_results(cwd=''):
    """
    Helper function to plot the error of the Hamiltonian

    Args:
        cwd (str): current working directory
    """

    # read in the dill data
    f = open(cwd + 'data/fput.dat', 'rb')
    stats = dill.load(f)
    f.close()

    plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()

    # HAMILTONIAN PLOTTING #
    # extract error in hamiltonian and prepare for plotting
    extract_stats = filter_stats(stats, type='err_hamiltonian')
    result = defaultdict(list)
    for k, v in extract_stats.items():
        result[k.iter].append((k.time, v))
    for k, v in result.items():
        result[k] = sorted(result[k], key=lambda x: x[0])

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    # Rearrange data for easy plotting
    err_ham = 1
    for k, v in result.items():
        time = [item[0] for item in v]
        ham = [item[1] for item in v]
        err_ham = ham[-1]
        plt_helper.plt.semilogy(time, ham, '-', lw=1, label='Iter ' + str(k))
    print(err_ham)
    # assert err_ham < 6E-10, 'Error in the Hamiltonian is too large, got %s' % err_ham

    plt_helper.plt.xlabel('Time')
    plt_helper.plt.ylabel('Error in Hamiltonian')
    plt_helper.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fname = 'data/fput_hamiltonian'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'

    # ENERGY PLOTTING #
    # extract error in hamiltonian and prepare for plotting
    extract_stats = filter_stats(stats, type='energy_step')
    result = sort_stats(extract_stats, sortby='time')

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    # Rearrange data for easy plotting
    for mode in result[0][1].keys():
        time = [item[0] for item in result]
        energy = [item[1][mode] for item in result]
        plt_helper.plt.plot(time, energy, label=str(mode) + 'th mode')

    plt_helper.plt.xlabel('Time')
    plt_helper.plt.ylabel('Energy')
    plt_helper.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fname = 'data/fput_energy'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'

    # POSITION PLOTTING #
    # extract positions and prepare for plotting
    extract_stats = filter_stats(stats, type='position')
    result = sort_stats(extract_stats, sortby='time')

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    # Rearrange data for easy plotting
    nparts = len(result[0][1])
    nsteps = len(result)
    pos = np.zeros((nparts, nsteps))
    time = np.zeros(nsteps)
    for idx, item in enumerate(result):
        time[idx] = item[0]
        for n in range(nparts):
            pos[n, idx] = item[1][n]

    for n in range(min(nparts, 16)):
        plt_helper.plt.plot(time, pos[n, :])

    fname = 'data/fput_positions'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def main():
    run_simulation()
    show_results()


if __name__ == "__main__":
    main()
