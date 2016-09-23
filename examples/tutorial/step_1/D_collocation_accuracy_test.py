import numpy as np
import scipy.sparse as sp
from collections import namedtuple

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh

from pySDC.collocation_classes.gauss_radau_right import CollGaussRadau_Right

def run_accuracy_test(prob, coll, dt_list):
    """
    Routine to build and solve the linear collocation problem

    Args:
        prob: a problem instance
        coll: a collocation instance
        dt: time-step size

    Return:
        the analytic error of the solved collocation problem
    """

    results = {}
    # loop over all nvars
    for dt in dt_list:
        # shrink collocation matrix: first line and column deals with initial value, not needed here
        Q = coll.Qmat[1:, 1:]

        # build system matrix M of collocation problem
        M = sp.eye(prob.nvars * coll.num_nodes) - dt * sp.kron(Q, prob.A)

        # get initial value at t0 = 0
        u0 = prob.u_exact(t=0)
        # fill in u0-vector as right-hand side for the collocation problem
        u0_coll = np.kron(np.ones(coll.num_nodes), u0.values)
        # get exact solution at Tend = dt
        uend = prob.u_exact(t=dt)

        # solve collocation problem directly
        u_coll = sp.linalg.spsolve(M, u0_coll)

        # compute error
        err = np.linalg.norm(u_coll[-prob.nvars:] - uend.values, np.inf)
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
    for i in range(1,len(dt_list)):

        # get ids
        id = ID(dt=dt_list[i])
        id_prev = ID(dt=dt_list[i-1])

        # compute order as log(prev_error/this_error)/log(this_dt/old_dt) <-- depends on the sorting of the list!
        order.append(np.log(results[id]/results[id_prev])/np.log(dt_list[i]/dt_list[i-1]))

    return order



if __name__ == "__main__":
    """
    A simple test program to compute the order of accuracy in time
    """

    # setup id for gathering the results (will sort by dt)
    ID = namedtuple('ID', 'dt')

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1      # diffusion coefficient
    problem_params['freq'] = 4      # frequency for the test value
    problem_params['nvars'] = 16383 # number of DOFs in space

    # instantiate problem
    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # instantiate collocation class, relative to the time interval [0,1]
    coll = CollGaussRadau_Right(num_nodes=3, tleft=0, tright=1)

    # assemble list of dt
    dt_list = [0.1/2**p for p in range(0,4)]

    # run accuracy test for all dt
    results = run_accuracy_test(prob=prob, coll=coll, dt_list=dt_list)

    # get order of accuracy
    order = get_accuracy_order(results)

    print(order, np.isclose(order,2*coll.num_nodes-1,rtol=0.4))

    assert all(np.isclose(order,2*coll.num_nodes-1,rtol=0.4))


