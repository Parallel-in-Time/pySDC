import pytest


def single_run(dt, Tend, num_nodes, quad_type, QI, useMPI):
    """
    Runs a single advection problem with certain parameters

    Args:
        dt (float): Step size
        Tend (float): Final time
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        QI (str): Preconditioner
        useMPI (bool): Whether or not to use MPI

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorWithinQ,
    )

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-10

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = QI
    sweeper_params['comm'] = comm

    problem_params = {'freq': 2, 'nvars': 2**9, 'c': 1.0, 'stencil_type': 'center', 'order': 6, 'bc': 'periodic'}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 99

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = LogGlobalErrorPostStep
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {EstimateExtrapolationErrorWithinQ: {}}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller


def multiple_runs(dts, **kwargs):
    """
    Make multiple runs of a specific problem and record vital error information

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes

    Returns:
        dict: Errors for multiple runs
        int: Order of the collocation problem
    """
    from pySDC.helpers.stats_helper import get_sorted

    res = {}

    for dt in dts:
        stats, controller = single_run(Tend=5.0 * dt, dt=dt, **kwargs)

        res[dt] = {}
        res[dt]['e_loc'] = max([me[1] for me in get_sorted(stats, type='e_global_post_step')])
        res[dt]['e_ex'] = max([me[1] for me in get_sorted(stats, type='error_extrapolation_estimate')])

    coll_order = controller.MS[0].levels[0].sweep.coll.order
    return res, coll_order


def check_order(dts, **kwargs):
    """
    Check the order by calling `multiple_runs` and then `plot_and_compute_order`.

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes
    """
    import numpy as np

    res, coll_order = multiple_runs(dts, **kwargs)
    dts = np.array(list(res.keys()))
    keys = list(res[dts[0]].keys())

    expected_order = {
        'e_loc': coll_order + 1,
        'e_ex': kwargs['num_nodes'] + 1,
    }

    for key in keys:
        errors = np.array([res[dt][key] for dt in dts])

        mask = np.logical_and(errors < 1e-0, errors > 1e-10)
        order = np.log(errors[mask][1:] / errors[mask][:-1]) / np.log(dts[mask][1:] / dts[mask][:-1])

        assert np.isclose(
            np.mean(order), expected_order[key], atol=0.5
        ), f'Expected order {expected_order[key]} for {key}, but got {np.mean(order):.2e}!'


@pytest.mark.base
@pytest.mark.parametrize('num_nodes', [2, 3])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_extrapolation_within_Q(num_nodes, quad_type):
    kwargs = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'useMPI': False,
        'QI': 'MIN',
    }
    check_order([5e-1, 1e-1, 8e-2, 5e-2], **kwargs)


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_nodes', [2, 3])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT'])
def test_extrapolation_within_Q_MPI(num_nodes, quad_type):
    import subprocess
    import os

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_nodes} python {__file__} {num_nodes} {quad_type}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_nodes,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        kwargs = {
            'num_nodes': int(sys.argv[1]),
            'quad_type': sys.argv[2],
            'useMPI': True,
            'QI': 'MIN',
        }
        check_order([5e-1, 1e-1, 8e-2, 5e-2], **kwargs)
