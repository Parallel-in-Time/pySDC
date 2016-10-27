import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC.implementations.sweeper_classes.linearized_implicit_parallel import linearized_implicit_parallel
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.01

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 1
    problem_params['nvars'] = 255
    problem_params['lambda0'] = 5.0
    problem_params['maxiter'] = 50
    problem_params['newton_tol'] = 1E-12
    problem_params['interval'] = (-5, 5)

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'
    sweeper_params['fixed_time_in_jacobian'] = 0

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = generalized_fisher
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    sweeper_list = [generic_implicit, linearized_implicit_fixed_parallel, linearized_implicit_parallel]

    f = open('parallelSDC_nonlinear_out.txt', 'w')
    uinit = None
    uex = None
    uend = None
    P = None

    # loop over the different sweepers and check results
    for sweeper in sweeper_list:
        description['sweeper_class'] = sweeper

        # instantiate the controller
        controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params,
                                                 description=description)

        # setup parameters "in time"
        t0 = 0
        Tend = 0.1

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err = abs(uex - uend)

        print('error at time %s: %s' % (Tend, err))

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        # compute and print statistics
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
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % \
              (float(np.std(niters)), float(np.var(niters)))
        f.write(out + '\n')
        f.write(out + '\n')
        print(out)

        f.write('\n')
        print()

        assert err < 3.68578e-05, 'ERROR: error is too high for sweeper %s, got %s' % (sweeper.__name__, err)
        assert np.mean(niters) == 7.5, 'ERROR: mean number of iterations not as expected, got %s' % np.mean(niters)

    f.close()

    plot_graphs(uinit, uend, uex, P)

    assert os.path.isfile('parallelSDC_fisher.png'), 'ERROR: plotting did not create file'


def plot_graphs(uinit, uend, uex, P):
    """
    Helper function to plot graphs of initial and final values

    Args:
        uinit: initial values
        uend: computed values
        uex: final values
        P: problem class
    """

    # Set up plotting parameters
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
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim((P.params.interval[0] - P.dx, P.params.interval[1] + P.dx))
    plt.ylim((-0.1, 1.1))
    plt.grid()

    # compute values for x-axis and plot
    xvalues = np.array([(i + 1 - (P.params.nvars + 1) / 2) * P.dx for i in range(P.params.nvars)])
    plt.plot(xvalues, uinit.values, 'r--', lw=2, label='initial')
    plt.plot(xvalues, uend.values, 'bs', lw=2, label='computed')
    plt.plot(xvalues, uex.values, 'gd', lw=2, label='exact')

    plt.legend(loc=2, ncol=1, numpoints=1)

    # save plot as PDF, beautify
    fname = 'parallelSDC_fisher.png'
    plt.savefig(fname, rasterized=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
