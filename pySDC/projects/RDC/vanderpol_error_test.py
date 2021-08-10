import matplotlib

matplotlib.use('Agg')

import matplotlib.pylab as plt

import numpy as np
import pickle
import os

from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.RDC.equidistant_RDC import Equidistant_RDC


def compute_RDC_errors():
    """
    Van der Pol's oscillator with RDC
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0
    level_params['dt'] = 10.0 / 40.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Equidistant_RDC
    sweeper_params['num_nodes'] = 41
    sweeper_params['QI'] = 'IE'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-14
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 10
    problem_params['u0'] = (2.0, 0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = None

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller_rdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # get initial values on finest level
    P = controller_rdc.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    ref_sol = np.load('data/vdp_ref.npy')

    maxiter_list = range(1, 11)
    results = dict()
    results['maxiter_list'] = maxiter_list

    for maxiter in maxiter_list:

        # ugly, but much faster than re-initializing the controller over and over again
        controller_rdc.MS[0].params.maxiter = maxiter

        # call main function to get things done...
        uend_rdc, stats_rdc = controller_rdc.run(u0=uinit, t0=t0, Tend=Tend)

        err = np.linalg.norm(uend_rdc - ref_sol, np.inf) / np.linalg.norm(ref_sol, np.inf)
        print('Maxiter = %2i -- Error: %8.4e' % (controller_rdc.MS[0].params.maxiter, err))
        results[maxiter] = err

    fname = 'data/vdp_results.pkl'
    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()

    assert os.path.isfile(fname), 'ERROR: pickle did not create file'


def plot_RDC_results(cwd=''):
    """
    Routine to visualize the errors

    Args:
        cwd (string): current working directory
    """

    file = open(cwd + 'data/vdp_results.pkl', 'rb')
    results = pickle.load(file, encoding='latin-1')
    file.close()

    # retrieve the list of nvars from results
    assert 'maxiter_list' in results, 'ERROR: expecting the list of maxiters in the results dictionary'
    maxiter_list = sorted(results['maxiter_list'])

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

    # create new figure
    plt.figure()
    # take x-axis limits from nvars_list + some spacning left and right
    plt.xlim([min(maxiter_list) - 1, max(maxiter_list) + 1])
    plt.xlabel('maxiter')
    plt.ylabel('rel. error')
    plt.grid()

    min_err = 1E99
    max_err = 0E00
    err_list = []
    # loop over nvars, get errors and find min/max error for y-axis limits
    for maxiter in maxiter_list:
        err = results[maxiter]
        min_err = min(err, min_err)
        max_err = max(err, max_err)
        err_list.append(err)
    plt.semilogy(maxiter_list, err_list, ls='-', marker='o', markersize=10, label='RDC')

    # adjust y-axis limits, add legend
    plt.ylim([min_err / 10, max_err * 10])
    plt.legend(loc=1, ncol=1, numpoints=1)

    # plt.show()

    # save plot as PNG, beautify
    fname = 'data/RDC_errors_vdp.png'
    plt.savefig(fname, bbox_inches='tight')

    assert os.path.isfile(fname), 'ERROR: plot was not created'

    return None


if __name__ == "__main__":
    compute_RDC_errors()
    plot_RDC_results()
