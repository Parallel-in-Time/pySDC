import matplotlib
matplotlib.use('Agg')

import numpy as np
from collections import namedtuple
import matplotlib.pylab as plt

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


def plot_accuracy(results):

    assert 'nvars_list' in results, 'ERROR: expecting the list of nvars in the results dictionary'

    nvars_list = sorted(results['nvars_list'])

    # Set up plotting parameters
    params = {'legend.fontsize': 20,
              'figure.figsize': (12, 8),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'lines.linewidth': 3
              }
    plt.rcParams.update(params)

    plt.figure()
    plt.xlim([min(nvars_list) / 2, max(nvars_list) * 2])
    plt.xlabel('nvars')
    plt.ylabel('abs. error')
    plt.grid()

    id = ID(nvars=nvars_list[0])
    base_error = results[id]
    order_guide_space = [base_error / (2 ** (2 * i)) for i in range(0, len(nvars_list))]
    plt.loglog(nvars_list, order_guide_space, color='k', ls='--', label='2nd order')

    min_err = 1E99
    max_err = 0E00
    err_list = []
    xvars = []
    for nvars in nvars_list:
        id = ID(nvars=nvars)
        err = results[id]
        min_err = min(err, min_err)
        max_err = max(err, max_err)
        err_list.append(err)
        xvars.append(nvars)

    plt.loglog(xvars, err_list, ls=' ', marker='o', markersize=10, label='experiment')

    plt.ylim([min_err / 10, max_err * 10])
    plt.legend(loc=1, ncol=1, numpoints=1)

    fname = 'accuracy_test.pdf'
    plt.savefig(fname, rasterized=True, bbox_inches='tight')

    return None


if __name__ == "__main__":

    ID = namedtuple('ID', 'nvars')

    problem_params = {}
    problem_params['nu'] = 0.1
    problem_params['freq'] = 4

    nvars_list = [2**p-1 for p in range(3,15)]

    results = run_accuracy_test(nvars_list=nvars_list)

    plot_accuracy(results)
