import numpy as np
from collections import namedtuple

from pySDC.implementations.problem_classes.HeatEquation_2D_FD_periodic import heat2d_periodic
from pySDC.implementations.datatype_classes.mesh import mesh

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars')


def test_spatial_accuracy():
    """
    A simple test program to check order of accuracy in space for a simple 2d test problem
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['freq'] = 2

    # create list of nvars to do the accuracy test with
    nvars_list = [(2 ** p, 2 ** p) for p in range(4, 12)]

    # run accuracy test for all nvars
    results = run_accuracy_check(nvars_list=nvars_list,problem_params=problem_params)

    # compute order of accuracy
    order = get_accuracy_order(results)

    assert (all(np.isclose(order, 2, rtol=0.005))), "ERROR: spatial order of accuracy is not as expected, got %s" %order


def run_accuracy_check(nvars_list,problem_params):
    """
    Routine to check the error of the Laplacian vs. its FD discretization

    Args:
        nvars_list: list of nvars to do the testing with
        problem_params: dictionary containing the problem-dependent parameters

    Returns:
        a dictionary containing the errors and a header (with nvars_list)
    """

    results = {}
    # loop over all nvars
    for nvars in nvars_list:

        # setup problem
        problem_params['nvars'] = nvars
        prob = heat2d_periodic(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        # create x values, use only inner points
        xvalues = np.array([i * prob.dx for i in range(prob.params.nvars[0])])

        # create a mesh instance and fill it with a sine wave
        u = prob.u_exact(t=0)

        # create a mesh instance and fill it with the Laplacian of the sine wave
        u_lap = prob.dtype_u(init=prob.init)
        u_lap.values = -2*(np.pi * prob.params.freq) ** 2 * prob.params.nu * np.kron(np.sin(np.pi * prob.params.freq * xvalues), np.sin(np.pi * prob.params.freq * xvalues))
        u_lap.values = u_lap.values.flatten()

        # compare analytic and computed solution using the eval_f routine of the problem class
        err = abs(prob.eval_f(u,0) - u_lap)

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
    for i in range(1,len(nvars_list)):

        # get ids
        id = ID(nvars=nvars_list[i])
        id_prev = ID(nvars=nvars_list[i-1])

        # compute order as log(prev_error/this_error)/log(this_nvars/old_nvars) <-- depends on the sorting of the list!
        order.append(np.log(results[id_prev]/results[id])/np.log(nvars_list[i][0]/nvars_list[i-1][0]))

    return order


# if __name__ == "__main__":
#     test_spatial_accuracy()
