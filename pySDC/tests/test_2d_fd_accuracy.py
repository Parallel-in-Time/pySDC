from collections import namedtuple

import numpy as np


# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars')


def test_spatial_accuracy():
    """
    A simple test program to check order of accuracy in space for a simple 2d test problem
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['freq'] = (2, 2)
    problem_params['nu'] = 1.0
    problem_params['bc'] = 'periodic'

    # create list of nvars to do the accuracy test with
    nvars_list = [(2**p, 2**p) for p in range(4, 12)]

    # run accuracy test for all nvars
    for order_stencil in [2, 4, 8]:
        results = run_accuracy_check(nvars_list=nvars_list, problem_params=problem_params, order_stencil=order_stencil)

        # compute order of accuracy
        order = get_accuracy_order(results)
        print(order_stencil, order)

        assert all(
            np.isclose(order, order_stencil, atol=5e-2)
        ), f"ERROR: expected spatial order to be {order_stencil} but got {np.mean(order):.2f}"


def run_accuracy_check(nvars_list, problem_params, order_stencil):
    """
    Routine to check the error of the Laplacian vs. its FD discretization

    Args:
        nvars_list: list of nvars to do the testing with
        problem_params: dictionary containing the problem-dependent parameters

    Returns:
        a dictionary containing the errors and a header (with nvars_list)
    """
    from pySDC.implementations.datatype_classes.mesh import mesh
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    results = {}
    # loop over all nvars
    for nvars in nvars_list:

        # setup problem
        problem_params['nvars'] = nvars
        problem_params['order'] = order_stencil
        prob = heatNd_unforced(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        # create x values, use only inner points
        xvalues = np.array([i * prob.dx for i in range(prob.params.nvars[0])])

        # create a mesh instance and fill it with a sine wave
        u = prob.u_exact(t=0)

        # create a mesh instance and fill it with the Laplacian of the sine wave
        u_lap = prob.dtype_u(init=prob.init)
        u_lap[:] = (
            -2
            * (np.pi**2 * prob.params.freq[0] * prob.params.freq[1])
            * prob.params.nu
            * np.kron(
                np.sin(np.pi * prob.params.freq[0] * xvalues), np.sin(np.pi * prob.params.freq[1] * xvalues)
            ).reshape(nvars)
        )
        # compare analytic and computed solution using the eval_f routine of the problem class
        err = abs(prob.eval_f(u, 0) - u_lap)

        # get id for this nvars and put error into dictionary
        id = ID(nvars=nvars)
        results[id] = err

    # add nvars_list to dictionary for easier access later on
    results['nvars_list'] = nvars_list

    return results


def get_accuracy_order(results):
    """
    Routine to compute the order of accuracy in space

    Args:
        results: the dictionary containing the errors

    Returns:
        the list of orders
    """

    # retrieve the list of nvars from results
    assert 'nvars_list' in results, 'ERROR: expecting the list of nvars in the results dictionary'
    nvars_list = sorted(results['nvars_list'])

    order = []
    # loop over two consecutive errors/nvars pairs
    for i in range(1, len(nvars_list)):

        # get ids
        id = ID(nvars=nvars_list[i])
        id_prev = ID(nvars=nvars_list[i - 1])

        # compute order as log(prev_error/this_error)/log(this_nvars/old_nvars) <-- depends on the sorting of the list!
        if results[id] > 1e-8 and results[id_prev] > 1e-8:
            order.append(np.log(results[id_prev] / results[id]) / np.log(nvars_list[i][0] / nvars_list[i - 1][0]))

    return order
