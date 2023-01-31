import pytest
import warnings
import numpy as np


@pytest.mark.base
def test_pendulum_u_exact_main():
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
    from pySDC.implementations.datatype_classes.mesh import mesh
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = pendulum_2d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # ignore using warning while checking error
    warnings.filterwarnings('ignore')
    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    # change warning status to error
    warnings.filterwarnings('error')
    try:
        u_test = prob.u_exact(5.0)
    except UserWarning:
        pass
    else:
        raise Exception("User warning not raised correctly")
    # reset warning status to normal
    warnings.resetwarnings()


@pytest.mark.base
def test_one_transistor_amplifier_u_exact_main():

    from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier
    from pySDC.implementations.datatype_classes.mesh import mesh
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = one_transistor_amplifier(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # ignore using warning while checking error
    warnings.filterwarnings('ignore')
    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    # change warning status to error
    warnings.filterwarnings('error')
    try:
        u_test = prob.u_exact(5.0)
    except UserWarning:
        pass
    else:
        raise Exception("User warning not raised correctly")
    # reset warning status to normal
    warnings.resetwarnings()


@pytest.mark.base
def test_two_transistor_amplifier_u_exact_main():
    
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier
    from pySDC.implementations.datatype_classes.mesh import mesh
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # instantiate problem
    prob = two_transistor_amplifier(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # ignore using warning while checking error
    warnings.filterwarnings('ignore')
    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(8))

    # change warning status to error
    warnings.filterwarnings('error')
    try:
        u_test = prob.u_exact(5.0)
    except UserWarning:
        pass
    else:
        raise Exception("User warning not raised correctly")
    # reset warning status to normal
    warnings.resetwarnings()


#
#   Explicit test for the pendulum example
#
@pytest.mark.base
def test_pendulum_main():
    
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 5e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 200

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = pendulum_2d
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error
    err = get_sorted(stats, type='error_post_step', sortby='time')
    err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_two_transistor_amplifier_main():
    
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-4

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = two_transistor_amplifier
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 2e-2

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # check error
    err = get_sorted(stats, type='error_post_step', sortby='time')
    err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    assert np.isclose(err, 0.0, atol=1e-3), "Error too large."
