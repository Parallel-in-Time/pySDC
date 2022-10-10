from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.LogisticEquation import logistics_equation
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.ODEs.trajectory_HookClass import trajectories


def main():
    """
    Van der Pol's oscillator inc. visualization
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12
    problem_params['newton_maxiter'] = 50
    problem_params['lam'] = 20
    problem_params['direct'] = True
    problem_params['u0'] = 0.5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = trajectories
    controller_params['logger_level'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = logistics_equation
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uex = P.u_exact(Tend)
    print(abs(uend - uex))


if __name__ == "__main__":
    main()
