from collections import namedtuple

import numpy as np

from implementations.datatype_classes.mesh import mesh
from implementations.problem_classes.HeatEquation_1D_FD import heat1d

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars')

def main():
    """
    A simple test program to check order of accuracy in space for a simple test problem
    """


    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value

    # create list of nvars to do the accuracy test with
    nvars_list = [2 ** p - 1 for p in range(4, 15)]

    # run accuracy test for all nvars
    results = run_accuracy_check(nvars_list=nvars_list,problem_params=problem_params)

    # compute order of accuracy
    order = get_accuracy_order(results)

    assert (all(np.isclose(order, 2, rtol=0.1))), "ERROR: spatial order of accuracy is not as expected, got %s" %order


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
        prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        # create x values, use only inner points
        xvalues = np.array([(i + 1) * prob.dx for i in range(prob.params.nvars)])

        # create a mesh instance and fill it with a sine wave
        u = prob.dtype_u(init=prob.init)
        u.values = np.sin(np.pi * prob.params.freq * xvalues)

        # create a mesh instance and fill it with the Laplacian of the sine wave
        u_lap = prob.dtype_u(init=prob.init)
        u_lap.values = -(np.pi * prob.params.freq) ** 2 * prob.params.nu * np.sin(np.pi * prob.params.freq * xvalues)

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
    for i in range(1,len(nvars_list)):

        # get ids
        id = ID(nvars=nvars_list[i])
        id_prev = ID(nvars=nvars_list[i-1])

        # compute order as log(prev_error/this_error)/log(this_nvars/old_nvars) <-- depends on the sorting of the list!
        order.append(np.log(results[id_prev]/results[id])/np.log(nvars_list[i]/nvars_list[i-1]))

    return order


if __name__ == "__main__":
    main()

