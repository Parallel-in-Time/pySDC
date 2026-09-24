from pathlib import Path
import matplotlib

matplotlib.use('Agg')

from collections import namedtuple
import matplotlib.pylab as plt
import numpy as np
import os.path
import scipy.sparse as sp

from pySDC.core.collocation import CollBase
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

# setup id for gathering the results (will sort by dt)
ID = namedtuple('ID', 'dt')


def main():
    """
    A simple test program to compute the order of accuracy in time
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 16383  # number of DOFs in space
    problem_params['bc'] = 'dirichlet-zero'  # boundary conditions

    # instantiate problem
    prob = heatNd_unforced(**problem_params)

    # instantiate collocation class, relative to the time interval [0,1]
    coll = CollBase(num_nodes=3, tleft=0, tright=1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')

    # assemble list of dt
    dt_list = [0.1 / 2**p for p in range(0, 5)]

    # run accuracy test for all dt
    results = run_accuracy_check(prob=prob, coll=coll, dt_list=dt_list)

    # get order of accuracy
    order = get_accuracy_order(results)

    # We solve a single step, so this is the local error, which for a collocation method of order 2M-1 is of order 2M.
    expected_order = 2 * coll.num_nodes

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_1_D_out.txt', 'w')
    for l in range(len(order)):
        out = 'Expected order: %2i -- Computed order %4.3f' % (expected_order, order[l])
        f.write(out + '\n')
        print(out)
    f.close()

    # visualize results
    plot_accuracy(results, order=expected_order)

    assert os.path.isfile('data/step_1_accuracy_test_coll.png')

    # The large dt are not asymptotic yet (|lambda dt| = 1.6 for the largest): the measured orders are 4.79, 5.36,
    # 5.68 and 5.88, approaching 6 from below. Only the last one is checked. Lobatto nodes, whose local order is 5,
    # get 4.86 there and would have passed the check this replaces, which accepted anything between 3 and 7.
    assert np.isclose(order[-1], expected_order, atol=0.3), (
        "ERROR: did not get order of accuracy as expected, got %s" % order
    )


def run_accuracy_check(prob, coll, dt_list):
    """
    Routine to build and solve the linear collocation problem

    Args:
        prob: a problem instance
        coll: a collocation instance
        dt_list: list of time-step sizes

    Return:
        the analytic error of the solved collocation problem
    """

    results = {}
    # loop over all nvars
    for dt in dt_list:
        # shrink collocation matrix: first line and column deals with initial value, not needed here
        Q = coll.Qmat[1:, 1:]

        # build system matrix M of collocation problem
        M = sp.eye(prob.nvars[0] * coll.num_nodes) - dt * sp.kron(Q, prob.A)

        # get initial value at t0 = 0
        u0 = prob.u_exact(t=0)
        # fill in u0-vector as right-hand side for the collocation problem
        u0_coll = np.kron(np.ones(coll.num_nodes), u0)
        # get exact solution at Tend = dt
        uend = prob.u_exact(t=dt)

        # solve collocation problem directly
        u_coll = sp.linalg.spsolve(M, u0_coll)

        # compute error
        err = np.linalg.norm(u_coll[-prob.nvars[0] :] - uend, np.inf)
        # get id for this dt and store error in results
        id = ID(dt=dt)
        results[id] = err

    # add list of dt to results for easier access
    results['dt_list'] = dt_list
    return results


def get_accuracy_order(results):
    """
    Routine to compute the order of accuracy in time

    Args:
        results: the dictionary containing the errors

    Returns:
        the list of orders
    """

    # retrieve the list of dt from results
    assert 'dt_list' in results, 'ERROR: expecting the list of dt in the results dictionary'
    dt_list = sorted(results['dt_list'], reverse=True)

    order = []
    # loop over two consecutive errors/dt pairs
    for i in range(1, len(dt_list)):
        # get ids
        id = ID(dt=dt_list[i])
        id_prev = ID(dt=dt_list[i - 1])

        # compute order as log(prev_error/this_error)/log(this_dt/old_dt) <-- depends on the sorting of the list!
        tmp = np.log(results[id] / results[id_prev]) / np.log(dt_list[i] / dt_list[i - 1])
        order.append(tmp)

    return order


def plot_accuracy(results, order):
    """
    Routine to visualize the errors as well as the expected errors

    Args:
        results: the dictionary containing the errors
        order (int): the expected order of accuracy, drawn as a guide
    """

    # retrieve the list of nvars from results
    assert 'dt_list' in results, 'ERROR: expecting the list of dts in the results dictionary'
    dt_list = sorted(results['dt_list'])

    # Set up plotting parameters
    params = {
        'legend.fontsize': 20,
        'figure.figsize': (12, 8),
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'lines.linewidth': 3,
    }
    plt.rcParams.update(params)

    # create new figure
    plt.figure()
    # take x-axis limits from nvars_list + some spacning left and right
    plt.xlim([min(dt_list) / 2, max(dt_list) * 2])
    plt.xlabel('dt')
    plt.ylabel('abs. error')
    plt.grid()

    # get guide for the order of accuracy, i.e. the errors to expect
    # get error for first entry in nvars_list
    id = ID(dt=dt_list[0])
    base_error = results[id]
    # assemble optimal errors for a method of the expected order and plot
    order_guide_space = [base_error * (2 ** (order * i)) for i in range(0, len(dt_list))]
    plt.loglog(dt_list, order_guide_space, color='k', ls='--', label=f'{order}th order')

    min_err = 1e99
    max_err = 0e00
    err_list = []
    # loop over nvars, get errors and find min/max error for y-axis limits
    for dt in dt_list:
        id = ID(dt=dt)
        err = results[id]
        min_err = min(err, min_err)
        max_err = max(err, max_err)
        err_list.append(err)
    plt.loglog(dt_list, err_list, ls=' ', marker='o', markersize=10, label='experiment')

    # adjust y-axis limits, add legend
    plt.ylim([min_err / 10, max_err * 10])
    plt.legend(loc=2, ncol=1, numpoints=1)

    # save plot as PDF, beautify
    fname = 'data/step_1_accuracy_test_coll.png'
    plt.savefig(fname, bbox_inches='tight')

    return None


if __name__ == "__main__":
    main()
