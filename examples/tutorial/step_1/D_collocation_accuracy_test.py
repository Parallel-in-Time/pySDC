import numpy as np
import scipy.sparse as sp
from collections import namedtuple

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh

from pySDC.collocation_classes.gauss_radau_right import CollGaussRadau_Right

def run_accuracy_test(prob, coll, dt_list):

    results = {}

    for dt in dt_list:

        Q = coll.Qmat[1:, 1:]

        M = sp.eye(prob.nvars * coll.num_nodes) - dt * sp.kron(Q, prob.A)

        u0 = prob.u_exact(t=0)
        uend = prob.u_exact(t=dt)
        u0_coll = np.kron(np.ones(coll.num_nodes), u0.values)

        u_coll = sp.linalg.spsolve(M, u0_coll)

        err = np.linalg.norm(u_coll[-prob.nvars:] - uend.values, np.inf)
        id = ID(dt=dt)
        results[id] = err

    results['dt_list'] = dt_list
    return results


def get_accuracy_order(results):

    assert 'dt_list' in results, 'ERROR: expecting the list of dt in the results dictionary'

    dt_list = sorted(results['dt_list'], reverse=True)

    order = []
    for i in range(1,len(dt_list)):

        id = ID(dt=dt_list[i])
        id_prev = ID(dt=dt_list[i-1])

        order.append(np.log(results[id]/results[id_prev])/np.log(dt_list[i]/dt_list[i-1]))

    return order



if __name__ == "__main__":

    ID = namedtuple('ID', 'dt')

    problem_params = {}
    problem_params['nu'] = 0.1
    problem_params['freq'] = 4
    problem_params['nvars'] = 16383

    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)
    coll = CollGaussRadau_Right(num_nodes=3, tleft=0, tright=1)

    dt_list = [0.1/2**p for p in range(0,4)]

    results = run_accuracy_test(prob=prob, coll=coll, dt_list=dt_list)

    order = get_accuracy_order(results)

    print(order, np.isclose(order,2*coll.num_nodes-1,rtol=0.4))

    assert all(np.isclose(order,2*coll.num_nodes-1,rtol=0.4))


