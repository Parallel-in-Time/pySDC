import pytest


@pytest.mark.base
def test_singularity():
    """
    Test if the singularity occurs at correct time.
    """

    import numpy as np
    from pySDC.implementations.problem_classes.nonlinear_ODE_1 import nonlinear_ODE_1

    problem_params = {
        'stop_at_nan': False,
    }

    nonlinear_ODE_class = nonlinear_ODE_1(**problem_params)
    t_event = 2

    u_event = nonlinear_ODE_class.u_exact(t_event)
    f = nonlinear_ODE_class.eval_f(u_event, t_event)

    assert f == 0, "Evaluation of right-hand side at singularity does not match with zero!"

    dt = 1e-1
    t0 = 1.9
    u0 = nonlinear_ODE_class.u_exact(t0)
    args = {
        'rhs': u0,
        'dt': dt,
        'u0': u0,
        't': t0,
    }

    sol = nonlinear_ODE_class.solve_system(**args)
    assert abs(sol - 1) < 1e-14, f"Solution is not close enough to the value at singularity! Expected 1, got {sol}"
    assert (
        nonlinear_ODE_class.newton_itercount == nonlinear_ODE_class.newton_maxiter
    ), f"Expected {nonlinear_ODE_class.newton_maxiter} Newton iterations, got {nonlinear_ODE_class.newton_itercount}"


@pytest.mark.base
def test_SDC_on_problem_class():
    """
    Test for SDC applied on problem class.
    """

    import numpy as np
    from pySDC.implementations.problem_classes.nonlinear_ODE_1 import nonlinear_ODE_1
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    level_params = {
        'restol': 1e-13,
        'dt': 1e-3,
    }

    sweeper_params = {
        'quad_type': 'LOBATTO',
        'num_nodes': 3,
        'QI': 'IE',
    }

    problem_params = {
        'stop_at_nan': False,
    }

    step_params = {
        'maxiter': 30,
    }

    controller_params = {
        'logger_level': 30,
    }

    description = dict()
    description['problem_class'] = nonlinear_ODE_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 1.999
    Tend = 2.0

    P = controller.MS[0].levels[0].prob
    uex = P.u_exact(Tend)

    uend, _ = controller.run(u0=uex, t0=t0, Tend=Tend)

    err = abs(uex - uend)
    assert err < 1.968e-8, f"Error is too large! Got {err}"
