import numpy as np
from collections import namedtuple

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh


def run_accuracy_test(nvars_list):

    results = {}
    for nvars in nvars_list:
        problem_params['nvars'] = nvars

        prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        xvalues = np.array([(i + 1) * prob.dx for i in range(prob.nvars)])

        u = prob.dtype_u(init=prob.nvars)
        u.values = np.sin(np.pi * prob.freq * xvalues)

        u_lap = prob.dtype_u(init=prob.nvars)
        u_lap.values = -(np.pi * prob.freq) ** 2 * prob.nu * np.sin(np.pi * prob.freq * xvalues)

        err = abs(prob.eval_f(u, 0) - u_lap)

        id = ID(nvars=nvars)
        results[id] = err

    results['nvars_list'] = nvars_list
    return results


def get_accuracy_order(results):

    assert 'nvars_list' in results, 'ERROR: expecting the list of nvars in the results dictionary'

    nvars_list = sorted(results['nvars_list'])

    order = []
    for i in range(1,len(nvars_list)):

        id = ID(nvars=nvars_list[i])
        id_prev = ID(nvars=nvars_list[i-1])

        order.append(np.log(results[id_prev]/results[id])/np.log(nvars_list[i]/nvars_list[i-1]))

    return order


if __name__ == "__main__":

    ID = namedtuple('ID', 'nvars')

    problem_params = {}
    problem_params['nu'] = 0.1
    problem_params['freq'] = 4

    nvars_list = [2**p-1 for p in range(3,15)]

    results = run_accuracy_test(nvars_list=nvars_list)

    order = get_accuracy_order(results)

    print(order, np.isclose(order,2,rtol=0.1))

    assert(all(np.isclose(order,2,rtol=0.1)))

