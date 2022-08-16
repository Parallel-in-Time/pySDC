import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


def main():
    """
    Van der Pol's oscillator reference solution
    """

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12
    level_params['dt'] = (Tend - t0) / 2000.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'IE'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-14
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 10
    problem_params['u0'] = (2.0, 0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller_ref = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller_ref.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend_ref, stats_ref = controller_ref.run(u0=uinit, t0=t0, Tend=Tend)

    np.save('data/vdp_ref.npy', uend_ref)


if __name__ == "__main__":
    main()
