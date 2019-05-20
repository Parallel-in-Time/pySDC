import os
import pickle

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    """
    Main driver running diffusion and advection tests
    """
    nsweeps = 3
    run_diffusion(nsweeps=nsweeps)
    run_advection(nsweeps=nsweeps)
    plot_results(nsweeps=nsweeps)

    nsweeps = 2
    run_diffusion(nsweeps=nsweeps)
    run_advection(nsweeps=nsweeps)
    plot_results(nsweeps=nsweeps)


def run_diffusion(nsweeps):
    """
    A simple test program to test PFASST convergence for the heat equation with random initial data

    Args:
        nsweeps: number of fine sweeps to perform
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [nsweeps, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['freq'] = -1  # frequency for the test value
    problem_params['nvars'] = [127, 63]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 4 * level_params['dt']

    # set up number of parallel time-steps to run PFASST with
    num_proc = 4

    results = dict()

    for i in range(-3, 10):
        ratio = level_params['dt'] / (1.0 / (problem_params['nvars'][0] + 1)) ** 2

        problem_params['nu'] = 10.0 ** i / ratio  # diffusion coefficient
        description['problem_params'] = problem_params  # pass problem parameters

        out = 'Working on c = %6.4e' % problem_params['nu']
        print(out)
        cfl = ratio * problem_params['nu']
        out = '  CFL number: %4.2e' % cfl
        print(out)

        # instantiate controller
        controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])

        out = '  Mean number of iterations: %4.2f' % np.mean(niters)
        print(out)

        if nsweeps == 3 and (i == -3 or i == 9):
            assert np.mean(niters) <= 2, 'ERROR: too much iterations for diffusive asymptotics, got %s' \
                                         % np.mean(niters)

        results[cfl] = np.mean(niters)

    fname = 'data/results_conv_diffusion_NS' + str(nsweeps) + '.pkl'
    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()

    assert os.path.isfile(fname), 'ERROR: pickle did not create file'


def run_advection(nsweeps):
    """
    A simple test program to test PFASST convergence for the periodic advection equation

    Args:
        nsweeps: number of fine sweeps to perform
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [nsweeps, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['freq'] = 64  # frequency for the test value
    problem_params['nvars'] = [128, 64]  # number of degrees of freedom for each level
    problem_params['order'] = 2
    problem_params['type'] = 'center'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advection1d  # pass problem class
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 4 * level_params['dt']

    # set up number of parallel time-steps to run PFASST with
    num_proc = 4

    results = dict()

    for i in range(-3, 10):
        ratio = level_params['dt'] / (1.0 / (problem_params['nvars'][0] + 1))

        problem_params['c'] = 10.0 ** i / ratio  # diffusion coefficient
        description['problem_params'] = problem_params  # pass problem parameters

        out = 'Working on nu = %6.4e' % problem_params['c']
        print(out)
        cfl = ratio * problem_params['c']
        out = '  CFL number: %4.2e' % cfl
        print(out)

        # instantiate controller
        controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])

        out = '  Mean number of iterations: %4.2f' % np.mean(niters)
        print(out)

        if nsweeps == 3 and (i == -3 or i == 9):
            assert np.mean(niters) <= 2, 'ERROR: too much iterations for advective asymptotics, got %s' \
                                         % np.mean(niters)

        results[cfl] = np.mean(niters)

    fname = 'data/results_conv_advection_NS' + str(nsweeps) + '.pkl'
    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()

    assert os.path.isfile(fname), 'ERROR: pickle did not create file'


def plot_results(nsweeps):
    """
    Plotting routine for iteration counts

    Args:
        nsweeps: number of fine sweeps used
    """

    fname = 'data/results_conv_diffusion_NS' + str(nsweeps) + '.pkl'
    file = open(fname, 'rb')
    results_diff = pickle.load(file)
    file.close()

    fname = 'data/results_conv_advection_NS' + str(nsweeps) + '.pkl'
    file = open(fname, 'rb')
    results_adv = pickle.load(file)
    file.close()

    xvalues_diff = sorted(list(results_diff.keys()))
    niter_diff = []
    for x in xvalues_diff:
        niter_diff.append(results_diff[x])

    xvalues_adv = sorted(list(results_adv.keys()))
    niter_adv = []
    for x in xvalues_adv:
        niter_adv.append(results_adv[x])

    # set up plotting parameters
    params = {'legend.fontsize': 20,
              'figure.figsize': (12, 8),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'lines.linewidth': 3
              }
    plt.rcParams.update(params)

    # set up figure
    plt.figure()
    plt.xlabel(r'$\mu$')
    plt.ylabel('no. of iterations')
    plt.xlim(min(xvalues_diff + xvalues_adv) / 10.0, max(xvalues_diff + xvalues_adv) * 10.0)
    plt.ylim(min(niter_diff + niter_adv) - 1, max(niter_diff + niter_adv) + 1)
    plt.grid()

    # plot
    plt.semilogx(xvalues_diff, niter_diff, 'r-', marker='s', markersize=10, label='diffusion')
    plt.semilogx(xvalues_adv, niter_adv, 'b-', marker='o', markersize=10, label='advection')

    plt.legend(loc=1, ncol=1, numpoints=1)

    # plt.show()
    # save plot, beautify
    fname = 'data/conv_test_niter_NS' + str(nsweeps) + '.png'
    plt.savefig(fname, rasterized=True, bbox_inches='tight')

    assert os.path.isfile(fname), 'ERROR: plotting did not create file'


if __name__ == "__main__":
    main()
