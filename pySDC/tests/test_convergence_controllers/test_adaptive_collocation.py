import pytest


def single_run(num_nodes, quad_type, QI, useMPI, params):
    """
    Runs a single advection problem with certain parameters

    Args:
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        QI (str): Preconditioner
        useMPI (bool): Whether or not to use MPI
        params (dict): Parameters for adaptive collocation convergence controller

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
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
    level_params['dt'] = 1.0
    level_params['restol'] = 1.0

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

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {AdaptiveCollocation: params}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    return controller


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
        'num_nodes': 3,
        'quad_type': 'RADAU-RIGHT',
        'QI': 'MIN',
        'useMPI': False,
        'params': coll_params_type,
        **kwargs,
    }

    # prepare variables
    controller = single_run(**args)
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
    step.status.active_coll = 0
    step.status.slot = 0
    level.u[0] = prob.u_exact(t=0)
    level.status.time = 0.0
    level.sweep.predict()
    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i])

    # perform the interpolation
    cont.switch_sweeper(controller.MS[0])
    step.status.active_coll = 1
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


def run_block(num_procs, num_nodes=[2, 3]):
    """
    Run one block of `num_procs` steps and report the collocation method each step ended on.

    Args:
        num_procs (int): number of steps in the block
        num_nodes (list): the collocation methods to walk through

    Returns:
        list: number of nodes each step finished with
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptive_collocation import AdaptiveCollocation
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    description = {
        'problem_class': polynomial_testequation,
        'problem_params': {'degree': max(num_nodes)},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': max(num_nodes), 'QI': 'LU'},
        'level_params': {'dt': 1.0, 'restol': 1e-8},
        'step_params': {'maxiter': 99},
        'convergence_controllers': {
            AdaptiveCollocation: {'num_nodes': num_nodes, 'quad_type': ['RADAU-RIGHT'] * len(num_nodes)}
        },
    }

    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    controller.run(u0=P.u_exact(0), t0=0.0, Tend=num_procs * 1.0)

    return [S.levels[0].sweep.coll.num_nodes for S in controller.MS]


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [1, 2, 4])
def test_all_steps_switch_collocation(num_procs):
    """
    Every step has to walk through all collocation methods, however many steps share a block.

    The index of the active collocation method used to live on the convergence controller, of which
    there is one per controller and therefore one per block. The first step to converge advanced it
    and switched itself, after which the guard stopped every other step from ever switching, so with
    `num_procs > 1` the collocation adaptivity was silently inert for all but one step.
    """
    num_nodes = [2, 3]
    nodes = run_block(num_procs, num_nodes=num_nodes)

    assert (
        nodes == [num_nodes[-1]] * num_procs
    ), f'Not all steps ended on the last collocation method: expected {[num_nodes[-1]] * num_procs}, got {nodes}'
