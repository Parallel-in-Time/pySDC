import numpy as np
from pathlib import Path

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced


def main():
    """
    A simple test program to set up a spatial problem and play with it
    """
    # instantiate problem
    prob = heatNd_unforced(
        nvars=1023,  # number of degrees of freedom
        nu=0.1,  # diffusion coefficient
        freq=4,  # frequency for the test value
        bc='dirichlet-zero'  # boundary conditions
        )

    # run accuracy test, get error back
    err = run_accuracy_check(prob)

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_1_A_out.txt', 'w')
    out = 'Error of the spatial accuracy test: %8.6e' % err
    f.write(out)
    print(out)
    f.close()

    assert err <= 2e-04, "ERROR: the spatial accuracy is higher than expected, got %s" % err


def run_accuracy_check(prob):
    """
    Routine to check the error of the Laplacian vs. its FD discretization

    Args:
        prob: a problem instance

    Returns:
        the error between the analytic Laplacian and the computed one of a given function
    """

    # create x values, use only inner points
    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.nvars[0])])

    # create a mesh instance and fill it with a sine wave
    u = prob.dtype_u(init=prob.init)
    u[:] = np.sin(np.pi * prob.freq[0] * xvalues)

    # create a mesh instance and fill it with the Laplacian of the sine wave
    u_lap = prob.dtype_u(init=prob.init)
    u_lap[:] = -((np.pi * prob.freq[0]) ** 2) * prob.nu * np.sin(np.pi * prob.freq[0] * xvalues)

    # compare analytic and computed solution using the eval_f routine of the problem class
    err = abs(prob.eval_f(u, 0) - u_lap)

    return err


if __name__ == "__main__":
    main()
