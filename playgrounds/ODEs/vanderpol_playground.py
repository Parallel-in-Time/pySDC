from pySDC_implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC_implementations.datatype_classes.mesh import mesh
from pySDC_implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC_implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC_implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from playgrounds.ODEs.trajectory_HookClass import trajectories

def main():
    """
    Van der Pol's oscillator inc. visualization
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-12
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 5
    problem_params['u0'] = (2.0, 0)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = trajectories
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
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 20.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)


if __name__ == "__main__":
    main()
