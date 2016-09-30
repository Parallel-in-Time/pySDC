from pySDC.Step import step

from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.generic_LU import generic_LU

from examples.tutorial.step_1.A_spatial_problem_setup import run_accuracy_check

def main():
    """
    A simple test program to setup a full step instance
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d               # pass problem class
    description['problem_params'] = problem_params      # pass problem parameters
    description['dtype_u'] = mesh                       # pass data type for u
    description['dtype_f'] = mesh                       # pass data type for f
    description['sweeper_class'] = generic_LU           # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params      # pass sweeper parameters
    description['level_params'] = level_params          # pass level parameters
    description['step_params'] = step_params            # pass step parameters

    # now the description contains more or less everything we need to create a step
    S = step(description=description)

    # we only have a single level, make a shortcut
    L = S.levels[0]

    # one of the integral parts of each level is the problem class, make a shortcut
    P = L.prob

    # now we can do e.g. what we did before with the problem
    err = run_accuracy_check(P)

    assert err <= 2E-04, "ERROR: the spatial accuracy is higher than expected, got %s" % err


if __name__ == "__main__":
    main()
