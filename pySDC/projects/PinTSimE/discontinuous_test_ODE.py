import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.DiscontinuousTestODE import DiscontinuousTestODE
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.PinTSimE.battery_model import runSimulation

import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_solution import LogSolution


class LogEventDiscontinuousTestODE(hooks):
    """
    Logs the problem dependent state function of the discontinuous test ODE.
    """

    def post_step(self, step, level_number):
        super(LogEventDiscontinuousTestODE, self).post_step(step, level_number)

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
    r"""
    Executes the simulation.

    Note
    ----
    Hardcoded solutions for battery models in `pySDC.projects.PinTSimE.hardcoded_solutions` are only computed for
    ``dt_list=[1e-2, 1e-3]`` and ``M_fix=4``. Hence changing ``dt_list`` and ``M_fix`` to different values could arise
    an ``AssertionError``.
    """

    # --- defines parameters for the problem class ----
    problem_params = {
        'newton_maxiter': 50,
        'newton_tol': 1e-11,
    }

    # --- defines parameters for sweeper ----
    M_fix = 3
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'LOBATTO',
        'QI': 'IE',
    }

    # --- defines parameters for event detection ----
    handling_params = {
        'restol': 1e-13,
        'maxiter': 8,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-12,
        'alpha': 1.0,
        'exact_event_time_avail': True,
    }

    # ---- all parameters are stored in this dictionary ----
    all_params = {
        'problem_params': problem_params,
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
    }

    hook_class = [LogEventDiscontinuousTestODE, LogSolution, LogGlobalErrorPostStep]

    use_detection = [True, False]
    use_adaptivity = [False]

    _ = runSimulation(
        problem=DiscontinuousTestODE,
        sweeper=generic_implicit,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(1.0, 2.0),
        dt_list=[1e-2, 1e-3],
        nnodes=[M_fix],
    )


if __name__ == "__main__":
    main()
