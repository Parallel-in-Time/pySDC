import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
from matplotlib.patches import Polygon

from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

from pySDC.core.Step import step


# noinspection PyShadowingNames
def compute_stability():
    """
    Routine to compute the stability domains of different configurations of fwsw-SDC

    Returns:
        numpy.ndarray: lambda_slow
        numpy.ndarray: lambda_fast
        int: number of collocation nodes
        int: number of iterations
        numpy.ndarray: stability numbers
    """
    N_s = 100
    N_f = 400

    lam_s_max = 5.0
    lam_f_max = 12.0
    lambda_s = 1j * np.linspace(0.0, lam_s_max, N_s)
    lambda_f = 1j * np.linspace(0.0, lam_f_max, N_f)

    problem_params = dict()
    # SET VALUE FOR lambda_slow AND VALUES FOR lambda_fast ###
    problem_params['lambda_s'] = np.array([0.0])
    problem_params['lambda_f'] = np.array([0.0])
    problem_params['u0'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    # SET TYPE AND NUMBER OF QUADRATURE NODES ###
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['do_coll_update'] = True

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = swfw_scalar  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = dict()  # pass step parameters

    # SET NUMBER OF ITERATIONS - SET K=0 FOR COLLOCATION SOLUTION ###
    K = 3

    # now the description contains more or less everything we need to create a step
    S = step(description=description)

    L = S.levels[0]

    Q = L.sweep.coll.Qmat[1:, 1:]
    nnodes = L.sweep.coll.num_nodes
    dt = L.params.dt

    stab = np.zeros((N_f, N_s), dtype='complex')

    for i in range(0, N_s):
        for j in range(0, N_f):
            lambda_fast = lambda_f[j]
            lambda_slow = lambda_s[i]
            if K is not 0:
                lambdas = [lambda_fast, lambda_slow]
                # LHS, RHS = L.sweep.get_scalar_problems_sweeper_mats(lambdas=lambdas)
                Mat_sweep = L.sweep.get_scalar_problems_manysweep_mat(nsweeps=K, lambdas=lambdas)
            else:
                # Compute stability function of collocation solution
                Mat_sweep = np.linalg.inv(np.eye(nnodes) - dt * (lambda_fast + lambda_slow) * Q)
            if L.sweep.params.do_coll_update:
                stab_fh = 1.0 + (lambda_fast + lambda_slow) * L.sweep.coll.weights.dot(
                    Mat_sweep.dot(np.ones(nnodes)))
            else:
                q = np.zeros(nnodes)
                q[nnodes - 1] = 1.0
                stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))
            stab[j, i] = stab_fh

    return lambda_s, lambda_f, sweeper_params['num_nodes'], K, stab


# noinspection PyShadowingNames
def plot_stability(lambda_s, lambda_f, num_nodes, K, stab):
    """
    Plotting routine of the stability domains

    Args:
        lambda_s (numpy.ndarray): lambda_slow
        lambda_f (numpy.ndarray): lambda_fast
        num_nodes (int): number of collocation nodes
        K (int): number of iterations
        stab (numpy.ndarray): stability numbers
    """

    lam_s_max = np.amax(lambda_s.imag)
    lam_f_max = np.amax(lambda_f.imag)

    rcParams['figure.figsize'] = 1.5, 1.5
    fs = 8
    fig = plt.figure()
    levels = np.array([0.25, 0.5, 0.75, 0.9, 1.1])
    CS1 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), levels, colors='k', linestyles='dashed')
    CS2 = plt.contour(lambda_s.imag, lambda_f.imag, np.absolute(stab), [1.0], colors='k')
    # Set markers at points used in plot_stab_vs_k
    plt.plot(4, 10, 'x', color='k', markersize=fs - 4)
    plt.plot(1, 10, 'x', color='k', markersize=fs - 4)
    plt.clabel(CS1, inline=True, fmt='%3.2f', fontsize=fs - 2)
    manual_locations = [(1.5, 2.5)]
    if K > 0:  # for K=0 and no 1.0 isoline, this crashes Matplotlib for somer reason
        plt.clabel(CS2, inline=True, fmt='%3.2f', fontsize=fs - 2, manual=manual_locations)
    plt.gca().add_patch(Polygon([[0, 0], [lam_s_max, 0], [lam_s_max, lam_s_max]], visible=True, fill=True,
                                facecolor='.75', edgecolor='k', linewidth=1.0, zorder=11))
    plt.gca().set_xticks(np.arange(0, int(lam_s_max) + 1))
    plt.gca().set_yticks(np.arange(0, int(lam_f_max) + 2, 2))
    plt.gca().tick_params(axis='both', which='both', labelsize=fs)
    plt.xlim([0.0, lam_s_max])
    plt.ylim([0.0, lam_f_max])
    plt.xlabel('$\Delta t \lambda_{slow}$', fontsize=fs, labelpad=0.0)
    plt.ylabel('$\Delta t \lambda_{fast}$', fontsize=fs, labelpad=0.0)
    plt.title(r'$M=%1i$, $K=%1i$' % (num_nodes, K), fontsize=fs)
    filename = 'data/stability-K' + str(K) + '-M' + str(num_nodes) + '.png'
    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    lambda_s, lambda_f, num_nodes, K, stab = compute_stability()
    plot_stability(lambda_s, lambda_f, num_nodes, K, stab)
