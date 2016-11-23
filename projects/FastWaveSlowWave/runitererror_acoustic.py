import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

from projects.FastWaveSlowWave.HookClass_acoustic import dump_energy
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.problem_classes.AcousticAdvection_1D_FD_imex import acoustic_1d_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.helpers.stats_helper import filter_stats


def compute_and_plot_itererror():
    """
    Routine to compute and plot the error over the iterations for difference cs values
    """

    num_procs = 1

    t0 = 0.0
    Tend = 0.025

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = Tend

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] = 15

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['cadv'] = 0.1
    problem_params['nvars'] = [(2, 300)]
    problem_params['order_adv'] = 5
    problem_params['waveno'] = 5

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['do_coll_update'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = acoustic_1d_imex
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['hook_class'] = dump_energy
    description['step_params'] = step_params
    description['level_params'] = level_params

    cs_v = [0.5, 1.0, 1.5, 5.0]
    nodes_v = [3]

    residual = np.zeros((np.size(cs_v), np.size(nodes_v), step_params['maxiter']))
    convrate = np.zeros((np.size(cs_v), np.size(nodes_v), step_params['maxiter'] - 1))
    lastiter = np.zeros((np.size(cs_v), np.size(nodes_v))) + step_params['maxiter']
    avg_convrate = np.zeros((np.size(cs_v), np.size(nodes_v)))

    P = None
    for cs_ind in range(0, np.size(cs_v)):
        problem_params['cs'] = cs_v[cs_ind]
        description['problem_params'] = problem_params

        for nodes_ind in np.arange(np.size(nodes_v)):

            sweeper_params['num_nodes'] = nodes_v[nodes_ind]
            description['sweeper_params'] = sweeper_params

            # instantiate the controller
            controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                                     description=description)
            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            print("Fast CFL number: %4.2f" % (problem_params['cs'] * level_params['dt'] / P.dx))
            print("Slow CFL number: %4.2f" % (problem_params['cadv'] * level_params['dt'] / P.dx))

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            extract_stats = filter_stats(stats, type='residual_post_iteration')

            for k, v in extract_stats.items():
                iter = getattr(k, 'iter')
                if iter is not -1:
                    residual[cs_ind, nodes_ind, iter - 1] = v

            # Compute convergence rates
            for iter in range(0, step_params['maxiter'] - 1):
                if residual[cs_ind, nodes_ind, iter] < level_params['restol']:
                    lastiter[cs_ind, nodes_ind] = iter
                else:
                    convrate[cs_ind, nodes_ind, iter] = residual[cs_ind, nodes_ind, iter + 1] / \
                        residual[cs_ind, nodes_ind, iter]
                avg_convrate[cs_ind, nodes_ind] = np.sum(convrate[cs_ind, nodes_ind, :]) / \
                    float(lastiter[cs_ind, nodes_ind])

    # Plot the results
    fs = 8
    color = ['r', 'b', 'g', 'c']
    shape = ['o-', 'd-', 's-', '>-']
    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    for ii in range(0, np.size(cs_v)):
        x = np.arange(1, lastiter[ii, 0])
        y = convrate[ii, 0, 0:int(lastiter[ii, 0]) - 1]
        plt.plot(x, y, shape[ii], markersize=fs - 2, color=color[ii],
                 label=r'$C_{\rm fast}$=%4.2f' % (cs_v[ii] * level_params['dt'] / P.dx))

    plt.legend(loc='upper right', fontsize=fs, prop={'size': fs - 2})
    plt.xlabel('Iteration', fontsize=fs)
    plt.ylabel(r'$|| r^{k+1} ||_{\infty}/|| r^k ||_{\infty}$', fontsize=fs, labelpad=2)
    plt.xlim([0, step_params['maxiter']])
    plt.ylim([0, 1.0])
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    filename = 'data/iteration.png'
    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    compute_and_plot_itererror()
