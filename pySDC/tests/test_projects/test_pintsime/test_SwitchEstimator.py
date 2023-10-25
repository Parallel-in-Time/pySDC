import numpy as np
import pytest

from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class ExactDiscontinuousTestODE(DiscontinuousTestODE):
    r"""
    Dummy problem for testing. The problem contains the exact dynamics of the problem class ``DiscontinuousTestODE``.
    """

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


def discontinuousTestProblem_run(problem, sweeper, quad_type, num_nodes, t0, Tend, dt, tol):
    """
    Simulates one run of a discontinuous test problem class for one specific tolerance specified in
    the parameters. For testing, only one time step should be considered.

    Parameters
    ----------
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
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
    from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

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
        'hook_class': LogSolution,
    }

    # convergence controllers
    convergence_controllers = dict()
    switch_estimator_params = {
        'tol': tol,
        'alpha': 1.0,
    }
    convergence_controllers.update({SwitchEstimator: switch_estimator_params})

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

    # set time parameters
    t0 = t0
    Tend = Tend

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    t_switch_exact = P.t_switch_exact

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


@pytest.mark.base
@pytest.mark.parametrize('tol', [10 ** (-m) for m in range(8, 13)])
@pytest.mark.parametrize('num_nodes', [3, 4, 5])
def test_all_tolerances_ODE(tol, num_nodes):
    """Tests a problem for different tolerances for the switch estimator."""
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.helpers.stats_helper import get_sorted

    t0 = 1.6
    Tend = 1.62
    problem = ExactDiscontinuousTestODE

    stats, t_switch_exact = discontinuousTestProblem_run(
        problem=problem,
        sweeper=generic_implicit,
        quad_type='LOBATTO',
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
