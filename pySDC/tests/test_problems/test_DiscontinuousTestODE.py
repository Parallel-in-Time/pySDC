import pytest


@pytest.mark.base
def test_event():
    """
    Test if the event occurs at correct time.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE

    problem_params = {
        'newton_tol': 1e-13,
    }

    DODE = DiscontinuousTestODE(**problem_params)

    u_event = DODE.u_exact(DODE.t_switch_exact)
    h = u_event[0] - 5

    assert abs(h) < 1e-15, 'Value of state function at exact event time is not zero!'

    t0 = 1.6
    dt = 1e-2
    u0 = DODE.u_exact(t0)

    args = {
        'rhs': u0,
        'dt': dt,
        'u0': u0,
        't': t0,
    }

    sol = DODE.solve_system(**args)
    assert np.isclose(
        sol[0], DODE.u_exact(t0 + dt)[0], atol=2e-3
    ), 'Solution after one time step is too far away from exact value!'


@pytest.mark.base
def test_capture_errors_and_warnings(caplog):
    r"""
    Test that checks if the errors in the problem classes are raised.
    """
    import numpy as np
    from pySDC.core.Errors import ProblemError
    from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE

    problem_params = {
        'newton_tol': 1e-13,
        'stop_at_nan': True,
    }

    DODE = DiscontinuousTestODE(**problem_params)

    t0 = 1.0
    dt = 1e-3
    u0 = DODE.u_exact(t0)

    args = {
        'rhs': u0,
        'dt': np.nan,
        'u0': u0,
        't': t0,
    }

    # test of ProblemError is raised
    with pytest.raises(ProblemError):
        DODE.solve_system(**args)

    # test if warnings are raised when nan values arises
    DODE.stop_at_nan = False
    DODE.solve_system(**args)
    assert 'Newton got nan after 100 iterations...' in caplog.text
    assert 'Newton did not converge after 100 iterations, error is nan' in caplog.text

    # test if warning is raised when local error is tried to computed
    u1 = DODE.u_exact(t0 + dt, u_init=u0, t_init=t0)
    assert (
        'DiscontinuousTestODE uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
        in caplog.text
    )
