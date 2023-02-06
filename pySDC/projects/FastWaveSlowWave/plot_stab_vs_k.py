import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
from matplotlib.ticker import ScalarFormatter

from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


from pySDC.core.Step import step


# noinspection PyShadowingNames
def compute_stab_vs_k(slow_resolved):
    """
    Routine to compute modulus of the stability function

    Args:
        slow_resolved (bool): switch to compute lambda_slow = 1 or lambda_slow = 4

    Returns:
        numpy.ndarray: number of nodes
        numpy.ndarray: number of iterations
        numpy.ndarray: moduli
    """

    mvals = [2, 3, 4]
    kvals = np.arange(1, 10)
    lambda_fast = 10j

    # PLOT EITHER FOR lambda_slow = 1 (resolved) OR lambda_slow = 4 (unresolved)
    if slow_resolved:
        lambda_slow = 1j
    else:
        lambda_slow = 4j
    stabval = np.zeros((np.size(mvals), np.size(kvals)))

    problem_params = dict()
    # SET VALUE FOR lambda_slow AND VALUES FOR lambda_fast ###
    problem_params['lambda_s'] = np.array([0.0])
    problem_params['lambda_f'] = np.array([0.0])
    problem_params['u0'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    # SET TYPE AND NUMBER OF QUADRATURE NODES ###
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['do_coll_update'] = True

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = swfw_scalar  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = dict()  # pass step parameters

    for i in range(0, np.size(mvals)):
        sweeper_params['num_nodes'] = mvals[i]
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters

        # now the description contains more or less everything we need to create a step
        S = step(description=description)

        L = S.levels[0]

        nnodes = L.sweep.coll.num_nodes

        for k in range(0, np.size(kvals)):
            Kmax = kvals[k]
            Mat_sweep = L.sweep.get_scalar_problems_manysweep_mat(nsweeps=Kmax, lambdas=[lambda_fast, lambda_slow])
            if L.sweep.params.do_coll_update:
                stab_fh = 1.0 + (lambda_fast + lambda_slow) * L.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))
            else:
                q = np.zeros(nnodes)
                q[nnodes - 1] = 1.0
                stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))
            stabval[i, k] = np.absolute(stab_fh)

    return mvals, kvals, stabval


# noinspection PyShadowingNames
def plot_stab_vs_k(slow_resolved, mvals, kvals, stabval):
    """
    Plotting routine for moduli

    Args:
        slow_resolved (bool): switch for lambda_slow
        mvals (numpy.ndarray): number of nodes
        kvals (numpy.ndarray): number of iterations
        stabval (numpy.ndarray): moduli
    """

    rcParams['figure.figsize'] = 2.5, 2.5
    fig = plt.figure()
    fs = 8
    plt.plot(kvals, stabval[0, :], 'o-', color='b', label=("M=%2i" % mvals[0]), markersize=fs - 2)
    plt.plot(kvals, stabval[1, :], 's-', color='r', label=("M=%2i" % mvals[1]), markersize=fs - 2)
    plt.plot(kvals, stabval[2, :], 'd-', color='g', label=("M=%2i" % mvals[2]), markersize=fs - 2)
    plt.plot(kvals, 1.0 + 0.0 * kvals, '--', color='k')
    plt.xlabel('Number of iterations K', fontsize=fs)
    plt.ylabel(r'Modulus of stability function $\left| R \right|$', fontsize=fs)
    plt.ylim([0.0, 1.2])
    if slow_resolved:
        plt.legend(loc='upper right', fontsize=fs, prop={'size': fs})
    else:
        plt.legend(loc='lower left', fontsize=fs, prop={'size': fs})

    plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    # plt.show()
    if slow_resolved:
        filename = 'data/stab_vs_k_resolved.png'
    else:
        filename = 'data/stab_vs_k_unresolved.png'

    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=True)
    print(np.amax(stabval))
    plot_stab_vs_k(True, mvals, kvals, stabval)
    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=False)
    print(np.amax(stabval))
    plot_stab_vs_k(False, mvals, kvals, stabval)
