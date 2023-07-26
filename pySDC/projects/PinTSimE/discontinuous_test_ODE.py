import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed, generate_description
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
from pySDC.implementations.hooks.log_solution import LogSolution


class LogEvent(hooks):
    """
    Logs the problem dependent state function of the discontinuous test ODE.
    """

    def post_step(self, step, level_number):
        super(LogEvent, self).post_step(step, level_number)

        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend[0] - 5,
        )


def main():
    """
    Executes the main stuff of the file.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [LogEvent, LogSolution, LogLocalErrorPostStep]

    problem_class = DiscontinuousTestODE

    sweeper = generic_implicit
    nnodes = [2, 3, 4]
    maxiter = 8
    newton_tol = 1e-11

    problem_params = dict()
    problem_params['newton_maxiter'] = 50
    problem_params['newton_tol'] = newton_tol

    use_detection = [True, False]

    t0 = 1.4
    Tend = 1.7
    dt_list = [1e-2, 1e-3]

    for dt in dt_list:
        for num_nodes in nnodes:
            for use_SE in use_detection:
                print(f'Controller run -- Simulation for step size: {dt}')

                restol = 1e-14
                recomputed = False if use_SE else None

                description, controller_params = generate_description(
                    dt,
                    problem_class,
                    sweeper,
                    num_nodes,
                    hookclass,
                    False,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    max_restarts=None,
                    tol_event=1e-10,
                )

                proof_assertions(description, t0, Tend, recomputed, use_SE)

                stats, t_switch_exact = controller_run(t0, Tend, controller_params, description)

                if use_SE:
                    switches = get_recomputed(stats, type='switch', sortby='time')
                    assert len(switches) >= 1, 'No events found!'
                    test_event_error(stats, dt, t_switch_exact, num_nodes)

                test_error(stats, dt, num_nodes, use_SE, recomputed)


def controller_run(t0, Tend, controller_params, description):
    """
    Executes a controller run for time interval to be specified in the arguments.

    Parameters
    ----------
    t0 : float
        Initial time of simulation.
    Tend : float
        End time of simulation.
    controller_params : dict
        Parameters needed for the controller.
    description : dict
        Contains all information for a controller run.

    Returns
    -------
    stats : dict
        Raw statistics from a controller run.
    """

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    t_switch_exact = P.t_switch_exact

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


def test_event_error(stats, dt, t_switch_exact, num_nodes):
    """
    Tests the error between the exact event time and the event time founded by switch estimation.

    The errors to the exact event time are very small. The higher the number of collocation nodes
    is the smaller the error to the exact event time is.

    Parameter
    ---------
    stats : dict
        Raw statistics from a controller run.
    dt : float
        Current step size.
    t_switch_exact : float
        Exact event time of the problem.
    num_nodes : int
        Number of collocation nodes used.
    """

    switches = get_recomputed(stats, type='switch', sortby='time')
    assert len(switches) >= 1, 'No switches found!'
    t_switches = [v[1] for v in switches]

    # dict with hardcoded solution for event time error
    t_event_err = {
        1e-2: {
            2: 1.7020665390443668e-06,
            3: 5.532252433937401e-10,
            4: 6.2776006615195e-11,
        },
        1e-3: {
            2: 1.7060500789867206e-08,
            3: 5.890081755666188e-10,
            4: 0.00057634035536136,
        },
    }

    t_event_err_got = abs(t_switch_exact - t_switches[-1])
    t_event_err_expected = t_event_err[dt][num_nodes]

    msg = f'Expected event time error {t_event_err_expected:.5f}, got {t_event_err_got:.5f}'
    assert np.isclose(t_event_err_got, t_event_err_expected, atol=5e-3), msg


def test_error(stats, dt, num_nodes, use_SE, recomputed):
    """
    Tests the error between the exact event solution and the numerical solution founded.

    In the dictionary containing the errors it can be clearly seen that errors are inherently reduced
    using the switch estimator to predict the event and adapt the time step to resolve the event in
    an more accurate way!

    Parameter
    ---------
    stats : dict
        Raw statistics from a controller run.
    dt : float
        Current step size.
    num_nodes : int
        Number of collocation nodes used.
    use_SE : bool
        Indicates whether switch detection is used or not.
    recomputed : bool
        Indicates whether the values after a restart will be used.
    """

    err = get_sorted(stats, type='e_local_post_step', sortby='time', recomputed=recomputed)
    err_norm = max([me[1] for me in err])

    u_err = {
        True: {
            1e-2: {
                2: 8.513409107457903e-06,
                3: 4.046930790480019e-09,
                4: 3.8459546658486943e-10,
            },
            1e-3: {
                2: 8.9185828500149e-08,
                3: 4.459276503609999e-09,
                4: 0.00015611434317808204,
            },
        },
        False: {
            1e-2: {
                2: 0.014137551021780936,
                3: 0.009855041165877765,
                4: 0.006289698596543047,
            },
            1e-3: {
                2: 0.002674332734426521,
                3: 0.00057634035536136,
                4: 0.00015611434317808204,
            },
        },
    }

    u_err_expected = u_err[use_SE][dt][num_nodes]
    u_err_got = err_norm

    msg = f'Expected event time error {u_err_expected:.7f}, got {u_err_got:.7f}'
    assert np.isclose(u_err_got - u_err_expected, 0, atol=1e-11), msg


def proof_assertions(description, t0, Tend, recomputed, use_detection):
    """
    Tests the parameters if they would not change.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    t0 : float
        Starting time.
    Tend : float
        End time.
    recomputed : bool
        Indicates whether the values after a restart are considered.
    use_detection : bool
        Indicates whether switch estimation is used.
    """

    newton_tol = description['problem_params']['newton_tol']
    msg = 'Newton tolerance should be set as small as possible to get best possible resolution of solution'
    assert newton_tol <= 1e-8, msg

    assert t0 >= 1.0, 'Problem is only defined for t >= 1!'
    assert Tend >= np.log(5), f'To investigate event, please set Tend larger than {np.log(5):.5f}'

    num_nodes = description['sweeper_params']['num_nodes']
    for M in [2, 3, 4]:
        if num_nodes not in [2, 3, 4]:
            assert num_nodes == M, f'Hardcoded solutions are only for M={M}!'

    sweeper = description['sweeper_class'].__name__
    assert sweeper == 'generic_implicit', 'Only generic_implicit sweeper is tested!'

    if use_detection:
        assert recomputed == False, 'Be aware that recomputed is set to False by using switch detection!'


if __name__ == "__main__":
    main()
