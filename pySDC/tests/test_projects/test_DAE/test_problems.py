import pytest
import warnings
import numpy as np


@pytest.mark.base
def test_pendulum_u_exact_main():
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = pendulum_2d(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    u_test = prob.u_exact(5.0)


@pytest.mark.base
def test_one_transistor_amplifier_u_exact_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # instantiate problem
    prob = one_transistor_amplifier(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(5))

    u_test = prob.u_exact(5.0)


@pytest.mark.base
def test_two_transistor_amplifier_u_exact_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # instantiate problem
    prob = two_transistor_amplifier(**problem_params)

    u_test = prob.u_exact(5.0)
    assert np.array_equal(u_test, np.zeros(8))

    u_test = prob.u_exact(5.0)


#
#   Explicit test for the pendulum example
#
@pytest.mark.base
def test_pendulum_main():
    from pySDC.projects.DAE.problems.simple_DAE import pendulum_2d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
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
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 200

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

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
    uend_ref = [0.98613917, -0.16592027, 0.29956023, 1.77825875, 4.82500525]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_one_transistor_amplifier_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import one_transistor_amplifier
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
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
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 5

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = one_transistor_amplifier
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

    uend_ref = [-0.02182035, 3.06674603, 2.89634691, 2.45212382, -2.69727238]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_two_transistor_amplifier_main():
    from pySDC.projects.DAE.problems.transistor_amplifier import two_transistor_amplifier
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
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
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 8

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

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

    uend_ref = [
        -5.52721527e-03,
        3.00630407e00,
        2.84974338e00,
        4.07588343e00,
        2.12960582e00,
        2.19430889e00,
        5.89240699e00,
        9.99531182e-02,
    ]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."


@pytest.mark.base
def test_synchgen_infinite_bus_main():
    from pySDC.projects.DAE.problems.synchronous_machine import synchronous_machine_infinite_bus
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 14

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['hook_class'] = error_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = synchronous_machine_infinite_bus
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uend_ref = [
        8.30823565e-01,
        -4.02584174e-01,
        1.16966755e00,
        9.47592808e-01,
        -3.68076863e-01,
        -3.87492326e-01,
        -7.77837831e-01,
        -1.67347611e-01,
        1.34810867e00,
        5.46223705e-04,
        1.29690691e-02,
        -8.00823474e-02,
        3.10281509e-01,
        9.94039645e-01,
    ]

    # check error
    err = np.linalg.norm(uend - uend_ref, np.inf)
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."
