import pytest


def single_run(dt, Tend, num_nodes, quad_type, QI, useMPI, params):
    """
    Runs a single advection problem with certain parameters

    Args:
        dt (float): Step size
        Tend (float): Final time
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        QI (str): Preconditioner
        useMPI (bool): Whether or not to use MPI
        params (dict): Parameters for adaptive collocation convergence controller

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
    from pySDC.implementations.convergence_controller_classes.adaptive_collocation import AdaptiveCollocation

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

    problem_params = {'degree': num_nodes}

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
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {AdaptiveCollocation: params}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    if Tend > 0:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    else:
        stats = {}
    return stats, controller


def single_test(**kwargs):
    """
    Run a single test where the solution is replaced by a polynomial and the nodes are changed.
    Because we know the polynomial going in, we can check if the interpolation based change was
    exact. If the solution is not a polynomial or a polynomial of higher degree then the number
    of nodes, the change in nodes does add some error, of course, but here it is on the order of
    machine precision.
    """
    import numpy as np

    coll_params_type = {
        'quad_type': ['GAUSS', 'RADAU-RIGHT'],
    }

    args = {
        'dt': 0.1,
        'Tend': 0.0,
        'num_nodes': 3,
        'quad_type': 'RADAU-RIGHT',
        'QI': 'MIN',
        'useMPI': False,
        'params': coll_params_type,
        **kwargs,
    }

    # prepare variables
    stats, controller = single_run(**args)
    step = controller.MS[0]
    level = step.levels[0]
    prob = level.prob
    cont = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == 'AdaptiveCollocation' for me in controller.convergence_controllers]
        ][0]
    ]
    nodes = np.append([0], level.sweep.coll.nodes)

    # initialize variables
    cont.status.active_coll = 0
    step.status.slot = 0
    level.u[0] = prob.u_exact(t=0)
    level.status.time = 0.0
    level.sweep.predict()
    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i])

    # perform the interpolation
    cont.switch_sweeper(controller.MS[0])
    cont.status.active_coll = 1
    cont.switch_sweeper(controller.MS[0])
    nodes = np.append([0], level.sweep.coll.nodes)
    error = max([abs(level.u[i] - prob.u_exact(nodes[i])) for i in range(len(level.u)) if level.u[i] is not None])
    assert error < 1e-15, f'Interpolation not exact!, Got {error}'
    print(f'Passed test with error {error}')

    diff = min([abs(level.u[0] - prob.u_exact(nodes[i])) for i in range(1, len(level.u)) if level.u[i] is not None])
    assert diff > 1e-15, 'Solution is constant!'


@pytest.mark.base
def test_adaptive_collocation():
    single_test()


@pytest.mark.mpi4py
def test_adaptive_collocation_MPI():
    import subprocess
    import os

    num_nodes = 3

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_nodes} python {__file__} MPI".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_nodes,
    )


if __name__ == "__main__":
    import sys

    kwargs = {}
    if len(sys.argv) > 1:
        kwargs = {
            'useMPI': True,
        }
    single_test(**kwargs)
