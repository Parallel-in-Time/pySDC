import numpy as np
import pytest

from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE


class ExactDiscontinuousTestODE(DiscontinuousTestODE):
    r"""
    Dummy ODE problem for testing. The problem contains the exact dynamics of the problem class ``DiscontinuousTestODE``.
    """

    def __init__(self, newton_maxiter=100, newton_tol=1e-8):
        """Initialization routine"""
        super().__init__(newton_maxiter, newton_tol)

    def eval_f(self, u, t):
        """
        Derivative.

        Parameters
        ----------
        u : dtype_u
            Exact value of u.
        t : float
            Time :math:`t`.

        Returns
        -------
        f : dtype_f
            Derivative.
        """

        f = self.dtype_f(self.init)

        t_switch = np.inf if self.t_switch is None else self.t_switch
        h = u[0] - 5
        if h >= 0 or t >= t_switch:
            f[:] = 1
        else:
            f[:] = np.exp(t)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Just return the exact solution...

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        return self.u_exact(t)


class ExactDiscontinuousTestDAE(DiscontinuousTestDAE):
    r"""
    Dummy DAE problem for testing. The problem contains the exact dynamics of the problem class ``DiscontinuousTestDAE``.
    """

    def __init__(self, newton_tol=1e-8):
        """Initialization routine"""
        super().__init__(newton_tol)

    def eval_f(self, u, du, t):
        r"""
        Returns the exact right-hand side of the problem. The first components in both
        cases of ``f`` are set to 1 to avoid getting a zero residual (enforced the sweeper to stop
        since "convergence is reached").

        Parameters
        ----------
        u : dtype_u
            Exact value of u.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Time :math:`t`.

        Returns
        -------
        f : dtype_f
            Right-hand side.
        """

        f = self.dtype_f(self.init)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * u[0] - 100
        f = self.dtype_f(self.init)

        if h >= 0 or t >= t_switch:
            f[:] = (
                1,
                0,
            )
        else:
            f[:] = (
                1,
                0,
            )
        return f

    def solve_system(self, impl_sys, u0, t, **kwargs):
        r"""
        Just returns the derivative of the exact solution.

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me[:] = (np.sinh(t), np.cosh(t))
        else:
            me[:] = (np.sinh(self.t_switch_exact), np.cosh(self.t_switch_exact))
        return me


def getTestDict():
    testDict = {
        0: {
            't': [0, 2],
            'p': [1, 5],
            'tMid': [1],
            'pMid': [3],
        },
        1: {
            't': [1, 2, 4, 5],
            'p': [6, 4, 3, 7],
            'tMid': [1.5, 3],
            'pMid': [5, 3.5],
        },
    }
    return testDict


def getParamsRun():
    r"""
    Returns parameters for conroller run that are used in each test.
    """
    restol = 1e-13
    alpha = 0.95
    maxiter = 1
    max_restarts = 3
    useA = False
    useSE = True
    exact_event_time_avail = True
    return restol, alpha, maxiter, max_restarts, useA, useSE, exact_event_time_avail


@pytest.mark.base
def testExactDummyProblems():
    r"""
    Test for dummy problems. The test verifies that the dummy problems exactly returns the dynamics of
    the parent class. ``eval_f`` of ``ExactDiscontinuousTestDAE`` is not tested here, since it only returns
    a random right-hand side to enforce the sweeper to do not stop to compute.
    """

    from pySDC.implementations.datatype_classes.mesh import mesh

    childODE = ExactDiscontinuousTestODE(**{})
    parentODE = DiscontinuousTestODE(**{})
    assert childODE.t_switch_exact == parentODE.t_switch_exact, f"Exact event times between classes does not match!"

    t0 = 1.0
    dt = 0.1
    u0 = parentODE.u_exact(t0)
    rhs = u0.copy()

    uSolve = childODE.solve_system(rhs, dt, u0, t0)
    uExact = parentODE.u_exact(t0)
    assert np.allclose(uSolve, uExact)

    # same test for event time
    tExactEventODE = parentODE.t_switch_exact
    dt = 0.1
    u0Event = parentODE.u_exact(tExactEventODE)
    rhsEvent = u0.copy()

    uSolveEvent = childODE.solve_system(rhsEvent, dt, u0Event, tExactEventODE)
    uExactEvent = parentODE.u_exact(tExactEventODE)
    assert np.allclose(uSolveEvent, uExactEvent)

    fExactOde = childODE.eval_f(u0, t0)
    fOde = parentODE.eval_f(u0, t0)
    assert np.allclose(fExactOde, fOde), f"Right-hand sides do not match!"

    fExactOdeEvent = childODE.eval_f(u0Event, tExactEventODE)
    fOdeEvent = parentODE.eval_f(u0Event, tExactEventODE)
    assert np.allclose(fExactOdeEvent, fOdeEvent), f"Right-hand sides at event do not match!"

    childDAE = ExactDiscontinuousTestDAE(**{})
    parentDAE = DiscontinuousTestDAE(**{})
    assert childDAE.t_switch_exact == parentDAE.t_switch_exact, f"Exact event times between classes does not match!"

    t0 = 0.0
    u0 = mesh(childDAE.init)
    duExactDAE = childDAE.solve_system(impl_sys=None, u0=u0, t=t0)
    duDAE = mesh(parentDAE.init)
    duDAE[:] = (np.sinh(t0), np.cosh(t0))
    assert np.allclose(duExactDAE, duDAE), f"Method solve_system of dummy problem do not return exact derivative!"

    tExactEventDAE = parentDAE.t_switch_exact
    u0Event = mesh(childDAE.init)
    duExactDAEEvent = childDAE.solve_system(impl_sys=None, u0=u0Event, t=tExactEventDAE)
    duDAEEvent = mesh(parentDAE.init)
    duDAEEvent[:] = (np.sinh(tExactEventDAE), np.cosh(tExactEventDAE))
    assert np.allclose(
        duExactDAEEvent, duDAEEvent
    ), f"Method solve_system of dummy problem do not return exact derivative at event!"


@pytest.mark.base
@pytest.mark.parametrize("key", [0, 1])
def testInterpolationValues(key):
    """
    Test for linear interpolation class in switch estimator. Linear interpolation is tested against
    values in testDict that contains values for interpolation computed by hand.
    Further, NumPy's routine interp is used as reference to have a second test on hand.
    """
    from pySDC.projects.PinTSimE.switch_estimator import LinearInterpolation

    testDict = getTestDict()
    testSet = testDict[key]
    t, p = testSet['t'], testSet['p']
    tMid, pMid = testSet['tMid'], testSet['pMid']
    LinearInterpolator = LinearInterpolation(t, p)

    for m in range(len(t)):
        assert LinearInterpolator.eval(t[m]) == p[m]
        assert np.interp(t[m], t, p) == LinearInterpolator.eval(t[m])

    for m in range(len(tMid)):
        assert LinearInterpolator.eval(tMid[m]) == pMid[m]
        assert np.interp(tMid[m], t, p) == LinearInterpolator.eval(tMid[m])


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['LOBATTO', 'RADAU-RIGHT'])
def testAdaptInterpolationInfo(quad_type):
    r"""
    Tests if the method ``adapt_interpolation_info`` does what it is supposed to do.

    - For ``quad_type='RADAU-RIGHT'``, the value at ``t0`` has to be added to the list of
      ``state_function``, since it is no collocation node but can be used for the interpolation
      anyway.

    - For ``quad_type='LOBATTO'``, the first value in ``state_function`` has to be removed
      since ``t0`` is also a collocation node here and the state function would include double values.

    Parameters
    ----------
    quad_type : str
        Type of quadrature used.
    """

    from pySDC.projects.PinTSimE.battery_model import generateDescription
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    problem = ExactDiscontinuousTestODE
    problem_params = dict()
    t0 = 1.6
    Tend = problem(**problem_params).t_switch_exact
    eps = 1e-13  # choose this eps to enforce a sign chance in state function
    dt = (Tend - t0) + eps

    sweeper = generic_implicit
    num_nodes = 3
    QI = 'IE'

    tol = 1e-10

    restol, alpha, maxiter, max_restarts, useA, useSE, _ = getParamsRun()

    hook_class = []

    _, _, controller = generateDescription(
        dt=dt,
        problem=problem,
        sweeper=sweeper,
        num_nodes=num_nodes,
        quad_type=quad_type,
        QI=QI,
        hook_class=hook_class,
        use_adaptivity=useA,
        use_switch_estimator=useSE,
        problem_params=problem_params,
        restol=restol,
        maxiter=maxiter,
        max_restarts=max_restarts,
        tol_event=tol,
        alpha=alpha,
    )

    S = controller.MS[0]
    L = S.levels[0]
    P = L.prob

    # instance of switch estimator
    SE = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == SwitchEstimator.__name__ for me in controller.convergence_controllers]
        ][0]
    ]

    S.status.slot = 0
    L.status.time = t0
    S.status.iter = 10
    L.status.residual = 0.0
    L.u[0] = P.u_exact(L.status.time)

    L.sweep.predict()

    # perform one update to get state function with different signs
    L.sweep.update_nodes()

    SE.get_new_step_size(controller, S)

    t_interp, state_function = SE.params.t_interp, SE.params.state_function

    assert len(t_interp) == len(
        state_function
    ), 'Length of interpolation values does not match with length of list containing the state function'

    if quad_type == 'LOBATTO':
        assert t_interp[0] != t_interp[1], 'Starting time from interpolation axis is not removed!'
        assert (
            len(t_interp) == num_nodes
        ), f'Number of values on interpolation axis does not match. Expected {num_nodes}, got {len(t_interp)}'

    elif quad_type == 'RADAU-RIGHT':
        assert (
            len(t_interp) == num_nodes + 1
        ), f'Number of values on interpolation axis does not match. Expected {num_nodes + 1}, got {len(t_interp)}'


@pytest.mark.base
@pytest.mark.parametrize('num_nodes', [3, 4, 5])
def testDetectionBoundary(num_nodes):
    """
    This test checks whether a restart is executed or not when the event exactly occurs at the boundary. In this case,
    no restart should be done because occuring the event at the boundary means that the event is already resolved well,
    i.e., the state function there should have a value close to zero.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    """

    from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_restarts import LogRestarts
    from pySDC.helpers.stats_helper import get_sorted

    problem = ExactDiscontinuousTestODE
    problem_params = dict()
    t0 = 1.6
    Tend = problem(**problem_params).t_switch_exact
    dt = Tend - t0

    sweeper = generic_implicit
    QI = 'IE'
    quad_type = 'LOBATTO'

    tol = 1e-10

    restol, alpha, maxiter, max_restarts, useA, useSE, exact_event_time_avail = getParamsRun()

    hook_class = [LogSolution, LogRestarts]

    description, controller_params, controller = generateDescription(
        dt=dt,
        problem=problem,
        sweeper=sweeper,
        num_nodes=num_nodes,
        quad_type=quad_type,
        QI=QI,
        hook_class=hook_class,
        use_adaptivity=useA,
        use_switch_estimator=useSE,
        problem_params=problem_params,
        restol=restol,
        maxiter=maxiter,
        max_restarts=max_restarts,
        tol_event=tol,
        alpha=alpha,
    )

    stats, _ = controllerRun(
        description=description,
        controller_params=controller_params,
        controller=controller,
        t0=t0,
        Tend=Tend,
        exact_event_time_avail=exact_event_time_avail,
    )

    sum_restarts = np.sum(np.array(get_sorted(stats, type='restart', sortby='time', recomputed=None))[:, 1])
    assert sum_restarts == 0, 'Event occurs at boundary, but restart(s) are executed anyway!'


@pytest.mark.base
@pytest.mark.parametrize('tol', [10 ** (-m) for m in range(8, 13)])
@pytest.mark.parametrize('num_nodes', [3, 4, 5])
@pytest.mark.parametrize('quad_type', ['LOBATTO', 'RADAU-RIGHT'])
def testDetectionODE(tol, num_nodes, quad_type):
    r"""
    Here, the switch estimator is applied to a dummy problem of ``DiscontinuousTestODE``,
    where the dynamics of the differential equation is replaced by its exact dynamics to see if
    the switch estimator predicts the event correctly. The problem is tested for a combination
    of different tolerances ``tol`` and different number of collocation nodes ``num_nodes``.

    Since the problem only uses the exact dynamics, the event should be predicted very accurately
    by the switch estimator.

    Parameters
    ----------
    tol : float
        Tolerance for switch estimator.
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Type of quadrature.
    """

    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_restarts import LogRestarts

    problem = ExactDiscontinuousTestODE
    problem_params = dict()
    t0 = 1.6
    Tend = 1.62
    dt = Tend - t0

    sweeper = generic_implicit
    QI = 'IE'

    restol, alpha, maxiter, max_restarts, useA, useSE, exact_event_time_avail = getParamsRun()

    hook_class = [LogSolution, LogRestarts]

    description, controller_params, controller = generateDescription(
        dt=dt,
        problem=problem,
        sweeper=sweeper,
        num_nodes=num_nodes,
        quad_type=quad_type,
        QI=QI,
        hook_class=hook_class,
        use_adaptivity=useA,
        use_switch_estimator=useSE,
        problem_params=problem_params,
        restol=restol,
        maxiter=maxiter,
        max_restarts=max_restarts,
        tol_event=tol,
        alpha=alpha,
    )

    stats, t_switch_exact = controllerRun(
        description=description,
        controller_params=controller_params,
        controller=controller,
        t0=t0,
        Tend=Tend,
        exact_event_time_avail=exact_event_time_avail,
    )

    # in this specific example only one event has to be found
    switches = [me[1] for me in get_sorted(stats, type='switch', sortby='time', recomputed=False)]
    assert len(switches) >= 1, f'{problem.__name__}: No events found for tol={tol}!'

    t_switch = switches[-1]
    event_err = abs(t_switch - t_switch_exact)
    assert np.isclose(event_err, 0, atol=1.2e-11), f'Event time error {event_err} is not small enough!'


@pytest.mark.base
@pytest.mark.parametrize('tol', [10 ** (-m) for m in range(8, 13)])
@pytest.mark.parametrize('num_nodes', [3, 4, 5])
def testDetectionDAE(tol, num_nodes):
    r"""
    In this test, the switch estimator is applied to a DAE dummy problem of ``DiscontinuousTestDAE``,
    where the dynamics of the differential equation is replaced by its exact dynamics to see if
    the switch estimator predicts the event correctly. The problem is tested for a combination
    of different tolerances ``tol`` and different number of collocation nodes ``num_nodes``.

    Since the problem only uses the exact dynamics, the event should be predicted very accurately
    by the switch estimator.

    Parameters
    ----------
    tol : float
        Tolerance for switch estimator.
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Type of quadrature.
    """

    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_restarts import LogRestarts

    problem = ExactDiscontinuousTestDAE
    problem_params = dict()
    t0 = 4.6
    Tend = 4.62
    dt = Tend - t0

    sweeper = fully_implicit_DAE
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'

    restol, alpha, maxiter, max_restarts, useA, useSE, exact_event_time_avail = getParamsRun()

    hook_class = [LogSolution, LogRestarts]

    description, controller_params, controller = generateDescription(
        dt=dt,
        problem=problem,
        sweeper=sweeper,
        num_nodes=num_nodes,
        quad_type=quad_type,
        QI=QI,
        hook_class=hook_class,
        use_adaptivity=useA,
        use_switch_estimator=useSE,
        problem_params=problem_params,
        restol=restol,
        maxiter=maxiter,
        max_restarts=max_restarts,
        tol_event=tol,
        alpha=alpha,
    )

    stats, t_switch_exact = controllerRun(
        description=description,
        controller_params=controller_params,
        controller=controller,
        t0=t0,
        Tend=Tend,
        exact_event_time_avail=exact_event_time_avail,
    )

    # in this specific example only one event has to be found
    switches = [me[1] for me in get_sorted(stats, type='switch', sortby='time', recomputed=False)]
    assert len(switches) >= 1, f'{problem.__name__}: No events found for tol={tol}!'

    t_switch = switches[-1]
    event_err = abs(t_switch - t_switch_exact)
    assert np.isclose(event_err, 0, atol=8e-14), f'Event time error {event_err} is not small enough!'
