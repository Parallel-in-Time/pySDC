import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from playgrounds.RDC.equidistant_RDC import Equidistant_RDC


def main():
    """
    Van der Pol's oscillator inc. visualization
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0
    level_params['dt'] = 10.0/40.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Equidistant_RDC
    sweeper_params['num_nodes'] = 41
    sweeper_params['QI'] = 'IE'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-14
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 10
    problem_params['u0'] = (2.0, 0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = None

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller_rdc = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params,
                                                 description=description)

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # get initial values on finest level
    P = controller_rdc.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    ref_sol = np.load('vdp_ref.npy')

    maxiter_list = range(1, 11)

    for maxiter in maxiter_list:

        # ugly, but much faster than re-initializing the controller over and over again
        controller_rdc.MS[0].params.maxiter = maxiter

        # call main function to get things done...
        uend_rdc, stats_rdc = controller_rdc.run(u0=uinit, t0=t0, Tend=Tend)

        err = np.linalg.norm(uend_rdc.values - ref_sol, np.inf) / np.linalg.norm(ref_sol, np.inf)
        print('Maxiter = %2i -- Error: %8.4e' % (controller_rdc.MS[0].params.maxiter, err))


if __name__ == "__main__":
    main()
