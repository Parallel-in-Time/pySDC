import pySDC.helpers.plot_helper as plt_helper

import pickle
import os

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_prec import linearized_implicit_fixed_parallel_prec
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.parallelSDC.GeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac
from pySDC.projects.parallelSDC.ErrReductionHook import err_reduction_hook

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 20

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 1
    problem_params['nvars'] = 2047
    problem_params['lambda0'] = 5.0
    problem_params['newton_maxiter'] = 50
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
    controller_params['hook_class'] = err_reduction_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = generalized_fisher_jac
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params

    # setup parameters "in time"
    t0 = 0
    Tend = 0.1

    sweeper_list = [generic_implicit, linearized_implicit_fixed_parallel, linearized_implicit_fixed_parallel_prec]
    dt_list = [Tend / 2 ** i for i in range(1, 5)]

    results = dict()
    results['sweeper_list'] = [sweeper.__name__ for sweeper in sweeper_list]
    results['dt_list'] = dt_list

    # loop over the different sweepers and check results
    for sweeper in sweeper_list:
        description['sweeper_class'] = sweeper
        error_reduction = []
        for dt in dt_list:
            print('Working with sweeper %s and dt = %s...' % (sweeper.__name__, dt))

            level_params['dt'] = dt
            description['level_params'] = level_params

            # instantiate the controller
            controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # filter statistics
            filtered_stats = filter_stats(stats, type='error_pre_iteration')
            error_pre = sort_stats(filtered_stats, sortby='iter')[0][1]

            filtered_stats = filter_stats(stats, type='error_post_iteration')
            error_post = sort_stats(filtered_stats, sortby='iter')[0][1]

            error_reduction.append(error_post / error_pre)

            print('error and reduction rate at time %s: %6.4e -- %6.4e' % (Tend, error_post, error_reduction[-1]))

        results[sweeper.__name__] = error_reduction
        print()

    file = open('data/error_reduction_data.pkl', 'wb')
    pickle.dump(results, file)
    file.close()


def plot_graphs(cwd=''):
    """
    Helper function to plot graphs of initial and final values

    Args:
        cwd (str): current working directory
    """
    plt_helper.mpl.style.use('classic')

    file = open(cwd + 'data/error_reduction_data.pkl', 'rb')
    results = pickle.load(file)

    sweeper_list = results['sweeper_list']
    dt_list = results['dt_list']

    color_list = ['red', 'blue', 'green']
    marker_list = ['o', 's', 'd']
    label_list = []
    for sweeper in sweeper_list:
        if sweeper == 'generic_implicit':
            label_list.append('SDC')
        elif sweeper == 'linearized_implicit_fixed_parallel':
            label_list.append('Simplified Newton')
        elif sweeper == 'linearized_implicit_fixed_parallel_prec':
            label_list.append('Inexact Newton')

    setups = zip(sweeper_list, color_list, marker_list, label_list)

    plt_helper.setup_mpl()

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    for sweeper, color, marker, label in setups:
        plt_helper.plt.loglog(dt_list, results[sweeper], lw=1, ls='-', color=color, marker=marker,
                              markeredgecolor='k', label=label)

    plt_helper.plt.loglog(dt_list, [dt * 2 for dt in dt_list], lw=0.5, ls='--', color='k', label='linear')
    plt_helper.plt.loglog(dt_list, [dt * dt / dt_list[0] * 2 for dt in dt_list], lw=0.5, ls='-.', color='k',
                          label='quadratic')

    plt_helper.plt.xlabel('dt')
    plt_helper.plt.ylabel('error reduction')
    plt_helper.plt.grid()

    # ax.set_xticks(dt_list, dt_list)
    plt_helper.plt.xticks(dt_list, dt_list)

    plt_helper.plt.legend(loc=1, ncol=1)

    plt_helper.plt.gca().invert_xaxis()
    plt_helper.plt.xlim([dt_list[0] * 1.1, dt_list[-1] / 1.1])
    plt_helper.plt.ylim([4E-03, 1E0])

    # save plot, beautify
    fname = 'data/parallelSDC_fisher_newton'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


if __name__ == "__main__":
    # main()
    plot_graphs()
