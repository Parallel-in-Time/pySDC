import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.Gander.HookClass_error_output import error_output


def testequation_setup(prec_type=None, maxiter=None):
    """
    Setup routine for the test equation

    Args:
        par (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0.0
    level_params['dt'] = 1.0
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = prec_type
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['u0'] = 1.0  # initial value (for all instances)
    # use single values like this...
    # problem_params['lambdas'] = [[-1.0]]
    # .. or a list of values like this ...
    # problem_params['lambdas'] = [[-1.0, -2.0, 1j, -1j]]
    problem_params['lambdas'] = [[-1.0 + 0j]]
    # note: PFASST will do all of those at once, but without interaction (realized via diagonal matrix).
    # The propagation matrix will be diagonal too, corresponding to the respective lambda value.

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = testequation0d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def compare_preconditioners(f=None, list_of_k=None):

    # set time parameters
    t0 = 0.0
    Tend = 2.0

    for k in list_of_k:

        description_IE, controller_params_IE = testequation_setup(prec_type='MIN3', maxiter=k)
        description_LU, controller_params_LU = testequation_setup(prec_type='LU', maxiter=k)

        out = f'\nWorking with maxiter = {k}'
        f.write(out + '\n')
        print(out)

        # instantiate controller
        controller_IE = controller_nonMPI(
            num_procs=1, controller_params=controller_params_IE, description=description_IE
        )
        controller_LU = controller_nonMPI(
            num_procs=1, controller_params=controller_params_LU, description=description_LU
        )

        # get initial values on finest level
        P = controller_IE.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        uex = P.u_exact(Tend)

        # this is where the iteration is happening
        uend_IE, stats_IE = controller_IE.run(u0=uinit, t0=t0, Tend=Tend)
        uend_LU, stats_LU = controller_LU.run(u0=uinit, t0=t0, Tend=Tend)

        diff = abs(uend_IE - uend_LU)

        err_IE = abs(uend_IE - uex)
        err_LU = abs(uend_LU - uex)

        out = '  Error (IE/LU) vs. exact solution: %6.4e -- %6.4e' % (err_IE, err_LU)
        f.write(out + '\n')
        print(out)
        out = '  Difference between both results: %6.4e' % diff
        f.write(out + '\n')
        print(out)

        # convert filtered statistics to list
        errors_IE = get_sorted(stats_IE, type='error_after_step', sortby='time')
        errors_LU = get_sorted(stats_LU, type='error_after_step', sortby='time')
        print(errors_IE)
        print(errors_LU)


def main():

    f = open('comparison_IE_vs_LU.txt', 'w')
    compare_preconditioners(f=f, list_of_k=[1])
    f.close()


if __name__ == "__main__":
    main()
