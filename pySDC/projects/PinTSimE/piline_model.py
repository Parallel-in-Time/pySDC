from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.projects.PinTSimE.battery_model import runSimulation

from pySDC.implementations.hooks.log_solution import LogSolution


def main():
    r"""
    Executes the simulation.

    Note
    ----
    Hardcoded solutions for battery models in `pySDC.projects.PinTSimE.hardcoded_solutions` are only computed for
    ``dt_list=[1e-2, 1e-3]`` and ``M_fix=4``. Hence changing ``dt_list`` and ``M_fix`` to different values could arise
    an ``AssertionError``.
    """

    # --- defines parameters for sweeper ----
    M_fix = 3
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'LOBATTO',
        'QI': 'IE',
    }

    # --- defines parameters for event detection ----
    handling_params = {
        'restol': 1e-12,
        'maxiter': 8,
        'max_restarts': None,
        'recomputed': None,
        'tol_event': None,
        'alpha': None,
        'exact_event_time_avail': None,
    }

    # ---- all parameters are stored in this dictionary - note: defaults are used for the problem ----
    all_params = {
        'problem_params': {},
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
    }

    hook_class = [LogSolution]

    use_detection = [False]
    use_adaptivity = [False]

    _ = runSimulation(
        problem=piline,
        sweeper=imex_1st_order,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(0.0, 15.0),
        dt_list=[5e-2, 1e-2],
        nnodes=[M_fix],
    )


if __name__ == "__main__":
    main()
