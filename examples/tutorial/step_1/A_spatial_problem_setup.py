import numpy as np

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh


def run_accuracy_test(prob):

    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.nvars)])

    u = prob.dtype_u(init=prob.nvars)
    u.values = np.sin(np.pi * prob.freq * xvalues)

    u_lap = prob.dtype_u(init=prob.nvars)
    u_lap.values = -(np.pi * prob.freq) ** 2 * prob.nu * np.sin(np.pi * prob.freq * xvalues)

    err = abs(prob.eval_f(u, 0) - u_lap)

    return err


if __name__ == "__main__":

    problem_params = {}
    problem_params['nu'] = 0.1
    problem_params['freq'] = 4
    problem_params['nvars'] = 1023

    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    err = run_accuracy_test(prob)

    print(err, err <= 2E-04)

    assert err <= 2E-04

