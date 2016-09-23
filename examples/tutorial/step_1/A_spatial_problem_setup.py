import numpy as np

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh


def run_accuracy_test(prob):
    """
    Routine to check the error of the Laplacian vs. its FD discretization

    Args:
        prob: a problem instance

    Returns:
        the error between the analytic Laplacian and the computed one of a given function
    """

    # create x values, use only inner points
    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.nvars)])

    # create a mesh instance and fill it with a sine wave
    u = prob.dtype_u(init=prob.nvars)
    u.values = np.sin(np.pi * prob.freq * xvalues)

    # create a mesh instance and fill it with the Laplacian of the sine wave
    u_lap = prob.dtype_u(init=prob.nvars)
    u_lap.values = -(np.pi * prob.freq) ** 2 * prob.nu * np.sin(np.pi * prob.freq * xvalues)

    # compare analytic and computed solution using the eval_f routine of the problem class
    err = abs(prob.eval_f(u, 0) - u_lap)

    return err


if __name__ == "__main__":
    """
    A simple test program to set up a spatial problem and play with it.
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1      # diffusion coefficient
    problem_params['freq'] = 4      # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # instantiate problem
    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # run accuracy test, get error back
    err = run_accuracy_test(prob)

    print(err, err <= 2E-04)

    assert err <= 2E-04

