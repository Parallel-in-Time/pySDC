import numpy as np
import pytest

from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class ExactDiscontinuousTestODE(DiscontinuousTestODE):
    r"""
    Dummy problem for testing. The problem contains the exact dynamics of the problem class ``DiscontinuousTestODE``.
    """

    def __init__(self, newton_maxiter=100, newton_tol=1e-8):
        """Initialization routine"""
        super().__init__(newton_maxiter, newton_tol)

        self.t_switch_exact = np.log(5)
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, t):
        """
        Derivative.

        Parameters
        ----------
        u : dtype_u
            Exact value of u.
        t : _type_
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


def get_controller(switch_estimator, problem, sweeper, quad_type, num_nodes, dt, tol):
    r"""
    This function prepares the controller for one simulation run. Based on function in
    ``pySDC.tests.test_convergence_controllers.test_adaptivity.py``

    Parameters
    ----------
    switch_estimator : pySDC.core.ConvergenceController
        Switch estimator.
    problem : pySDC.core.Problem
        Problem class to be simulated.
    sweeper: pySDC.core.Sweeper
        Sweeper used for the simulation.
    quad_type : str
        Type of quadrature used.
    num_nodes : int
        Number of collocation nodes.
    t0 : float
        Starting time.
    Tend : float
        End time.
    dt : float
        Time step size.
    tol : float
        Tolerance for the switch estimator.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_restarts import LogRestarts

    # initialize level parameters
    level_params = {
        'restol': 1e-13,
        'dt': dt,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': num_nodes,
        'QI': 'IE',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 10,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution, LogRestarts],
    }

    # convergence controllers
    convergence_controllers = dict()
    switch_estimator_params = {
        'tol': tol,
        'alpha': 1.0,
    }
    convergence_controllers.update({switch_estimator: switch_estimator_params})

    convergence_controllers[BasicRestartingNonMPI] = {
        'max_restarts': 3,
        'crash_after_max_restarts': False,
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,  # pass problem class
        'problem_params': dict(),  # problem_params,  # pass problem parameters
        'sweeper_class': sweeper,  # pass sweeper
        'sweeper_params': sweeper_params,  # pass sweeper parameters
        'level_params': level_params,  # pass level parameters
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    return controller


def discontinuousTestProblem_run(switch_estimator, problem, sweeper, quad_type, num_nodes, t0, Tend, dt, tol):
    """
    Simulates one run of a discontinuous test problem class for one specific tolerance specified in
    the parameters. For testing, only one time step should be considered.

    Parameters
    ----------
    switch_estimator : pySDC.core.ConvergenceController
        Switch Estimator.
    problem : pySDC.core.Problem
        Problem class to be simulated.
    sweeper: pySDC.core.Sweeper
        Sweeper used for the simulation.
    quad_type : str
        Type of quadrature used.
    num_nodes : int
        Number of collocation nodes.
    t0 : float
        Starting time.
    Tend : float
        End time.
    dt : float
        Time step size.
    tol : float
        Tolerance for the switch estimator.
    """

    controller = get_controller(switch_estimator, problem, sweeper, quad_type, num_nodes, dt, tol)

    # set time parameters
    t0 = t0
    Tend = Tend

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    t_switch_exact = P.t_switch_exact

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


@pytest.mark.base
@pytest.mark.parametrize('quad_type', ['LOBATTO', 'RADAU-RIGHT'])
def test_adapt_interpolation_info(quad_type):
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

    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    t0 = 1.6
    Tend = ExactDiscontinuousTestODE().t_switch_exact
    eps = 1e-13  # choose this eps to enforce a sign chance in state function
    dt = (Tend - t0) + eps

    tol = 1e-10
    num_nodes = 3

    controller = get_controller(
        switch_estimator=SwitchEstimator,
        problem=ExactDiscontinuousTestODE,
        sweeper=generic_implicit,
        quad_type=quad_type,
        num_nodes=num_nodes,
        dt=dt,
        tol=tol,
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
def test_detection_at_boundary(num_nodes):
    """
    This test checks whether a restart is executed or not when the event exactly occurs at the boundary. In this case,
    no restart should be done because occuring the event at the boundary means that the event is already resolved well,
    i.e., the state function there should have a value close to zero.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    """

    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.helpers.stats_helper import get_sorted

    problem = ExactDiscontinuousTestODE
    t0 = 1.6
    Tend = ExactDiscontinuousTestODE().t_switch_exact
    dt = Tend - t0

    stats, _ = discontinuousTestProblem_run(
        switch_estimator=SwitchEstimator,
        problem=problem,
        sweeper=generic_implicit,
        quad_type='LOBATTO',
        num_nodes=num_nodes,
        t0=t0,
        Tend=Tend,
        dt=dt,
        tol=1e-10,
    )

    sum_restarts = np.sum(np.array(get_sorted(stats, type='restart', sortby='time', recomputed=None))[:, 1])
    assert sum_restarts == 0, 'Event occurs at boundary, but restart(s) are executed anyway!'


@pytest.mark.base
@pytest.mark.parametrize('tol', [10 ** (-m) for m in range(8, 13)])
@pytest.mark.parametrize('num_nodes', [3, 4, 5])
@pytest.mark.parametrize('quad_type', ['LOBATTO', 'RADAU-RIGHT'])
def test_all_tolerances_ODE(tol, num_nodes, quad_type):
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
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.helpers.stats_helper import get_sorted

    t0 = 1.6
    Tend = 1.62
    problem = ExactDiscontinuousTestODE

    stats, t_switch_exact = discontinuousTestProblem_run(
        switch_estimator=SwitchEstimator,
        problem=problem,
        sweeper=generic_implicit,
        quad_type=quad_type,
        num_nodes=num_nodes,
        t0=t0,
        Tend=Tend,
        dt=2e-2,
        tol=tol,
    )

    # in this specific example only one event has to be found
    switches = [me[1] for me in get_sorted(stats, type='switch', sortby='time', recomputed=False)]
    assert len(switches) >= 1, f'{problem.__name__}: No events found for tol={tol}!'

    t_switch = switches[-1]
    event_err = abs(t_switch - t_switch_exact)
    assert event_err < 1e-14, f'Event time error for is not small enough!'