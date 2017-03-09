import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    """
    Main driver running diffusion and advection tests
    """
    QI = 'LU'
    # run_diffusion(QI=QI)
    run_advection(QI=QI)

    QI = 'LU2'
    # run_diffusion(QI=QI)
    run_advection(QI=QI)

    plot_results()


def run_diffusion(QI):
    """
    A simple test program to test PFASST convergence for the heat equation with random initial data

    Args:
        QI: preconditioner
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = [QI, 'LU']
    sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = -1  # frequency for the test value
    problem_params['nvars'] = [127, 63]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 200

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up number of parallel time-steps to run PFASST with

    fname = 'data/results_conv_diffusion_Linf_QI' + str(QI) + '.txt'
    file = open(fname, 'w')
    writer = csv.writer(file)
    writer.writerow(('num_proc', 'niter'))
    file.close()

    for i in range(0, 13):

        num_proc = 2 ** i
        level_params['dt'] = (Tend - t0) / num_proc
        description['level_params'] = level_params  # pass level parameters

        out = 'Working on num_proc = %5i' % num_proc
        print(out)
        cfl = problem_params['nu'] * level_params['dt'] / (1.0 / (problem_params['nvars'][0] + 1)) ** 2
        out = '  CFL number: %4.2e' % cfl
        print(out)

        # instantiate controller
        controller = allinclusive_multigrid_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                                   description=description)

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

        file = open(fname, 'a')
        writer = csv.writer(file)
        writer.writerow((num_proc, np.mean(niters)))
        file.close()

    assert os.path.isfile(fname), 'ERROR: pickle did not create file'


def run_advection(QI):
    """
    A simple test program to test PFASST convergence for the periodic advection equation

    Args:
        QI: preconditioner
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = [QI, 'LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['freq'] = 64  # frequency for the test value
    problem_params['nvars'] = [128, 64]  # number of degrees of freedom for each level
    problem_params['order'] = 2
    problem_params['type'] = 'center'
    problem_params['c'] = 0.1

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 200

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advection1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up number of parallel time-steps to run PFASST with

    fname = 'data/results_conv_advection_Linf_QI' + str(QI) + '.txt'
    file = open(fname, 'w')
    writer = csv.writer(file)
    writer.writerow(('num_proc', 'niter'))
    file.close()

    for i in range(0, 7):

        num_proc = 2 ** i
        level_params['dt'] = (Tend - t0) / num_proc
        description['level_params'] = level_params  # pass level parameters

        out = 'Working on num_proc = %5i' % num_proc
        print(out)
        cfl = problem_params['c'] * level_params['dt'] / (1.0 / problem_params['nvars'][0])
        out = '  CFL number: %4.2e' % cfl
        print(out)

        # instantiate controller
        controller = allinclusive_multigrid_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                                   description=description)

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

        file = open(fname, 'a')
        writer = csv.writer(file)
        writer.writerow((num_proc, np.mean(niters)))
        file.close()

    assert os.path.isfile(fname), 'ERROR: pickle did not create file'


def plot_results(cwd=''):
    """
    Plotting routine for iteration counts

    Args:
        cwd: current working directory
    """

    setups = [('diffusion', 'LU', 'LU2'), ('advection', 'LU', 'LU2')]

    for type, QI1, QI2 in setups:

        fname = cwd + 'data/results_conv_' + type + '_Linf_QI' + QI1 + '.txt'
        file = open(fname, 'r')
        reader = csv.DictReader(file, delimiter=',')
        xvalues_1 = []
        niter_1 = []
        for row in reader:
            xvalues_1.append(int(row['num_proc']))
            niter_1.append(float(row['niter']))
        file.close()

        fname = cwd + 'data/results_conv_' + type + '_Linf_QI' + QI2 + '.txt'
        file = open(fname, 'r')
        reader = csv.DictReader(file, delimiter=',')
        xvalues_2 = []
        niter_2 = []
        for row in reader:
            xvalues_2.append(int(row['num_proc']))
            niter_2.append(float(row['niter']))
        file.close()

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
        plt.xlabel('number of time-steps (L)')
        plt.ylabel('#iterations')
        plt.xlim(min(xvalues_1 + xvalues_2) / 2, max(xvalues_1 + xvalues_2) * 2)
        plt.ylim(min(niter_1 + niter_2) - 1, max(niter_1 + niter_2) + 1)
        plt.grid()

        # plot
        plt.semilogx(xvalues_1, niter_1, 'r-', marker='s', markersize=10, label=QI1)
        plt.semilogx(xvalues_2, niter_2, 'b-', marker='o', markersize=10, label=QI2)

        plt.legend(loc=2, ncol=1, numpoints=1)

        # save plot, beautify
        fname = 'data/conv_test_niter_Linf_' + type + '.png'
        plt.savefig(fname, rasterized=True, bbox_inches='tight')

        assert os.path.isfile(fname), 'ERROR: plotting did not create file'


if __name__ == "__main__":
    main()
