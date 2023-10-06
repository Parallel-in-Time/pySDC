import pytest


def get_controller(MPIsweeper, MPIcontroller):
    """
    Runs a single advection problem with certain parameters

    Args:
        MPIsweeper (bool): Use MPI parallel sweeper
        MPIcontroller (bool): Use MPI parallel controller

    Returns:
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.convergence_controller_classes.stop_at_nan import StopAtNan

    if MPIcontroller:
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI as controller_class
        from mpi4py import MPI

        controller_args = {'comm': MPI.COMM_WORLD}
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI as controller_class

        controller_args = {'num_procs': 1}

    if MPIsweeper:
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
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['comm'] = comm

    problem_params = {'degree': 12}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 0

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {StopAtNan: {'thresh': 1e3}}

    controller = controller_class(controller_params=controller_params, description=description, **controller_args)
    return controller


def single_test(MPIsweeper=False, MPIcontroller=False):
    """
    Run a single test where the solution is replaced by a polynomial and the nodes are changed.
    Because we know the polynomial going in, we can check if the interpolation based change was
    exact. If the solution is not a polynomial or a polynomial of higher degree then the number
    of nodes, the change in nodes does add some error, of course, but here it is on the order of
    machine precision.
    """
    import numpy as np
    from pySDC.core.Errors import ConvergenceError

    args = {
        'MPIsweeper': MPIsweeper,
        'MPIcontroller': MPIcontroller,
    }

    # prepare variables
    controller = get_controller(**args)

    if MPIcontroller:
        step = controller.S
        modify = controller.comm.rank == 0
        comm = controller.comm
    else:
        step = controller.MS[0]
        comm = None
        modify = True
    level = step.levels[0]
    prob = level.prob
    cont = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == 'StopAtNan' for me in controller.convergence_controllers]
        ][0]
    ]

    nodes = np.append([0], level.sweep.coll.nodes)

    # initialize variables
    step.status.slot = 0
    step.status.iter = 1
    level.status.time = 0.0
    level.status.residual = 0.0
    level.u[0] = prob.u_exact(t=0)
    level.sweep.predict()

    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i] * level.dt)

    cont.post_iteration_processing(controller, step, comm=comm)

    try:
        if modify:
            level.u[0][:] = np.nan
        cont.post_iteration_processing(controller, step, comm=comm)
        raise Exception('Did not raise error!')
    except ConvergenceError:
        print('Successfully raised error when nan is part of the solution')

    try:
        if modify:
            level.u[0][:] = 1e99
        cont.post_iteration_processing(controller, step, comm=comm)
        raise Exception('Did not raise error!')
    except ConvergenceError:
        print('Successfully raised error solution exceeds limit')


@pytest.mark.base
def test_stop_at_nan():
    single_test()


@pytest.mark.mpi4py
@pytest.mark.parametrize('mode', ['0 1', '1 0'])
def test_interpolation_error_MPI(mode):
    import subprocess
    import os

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {3} python {__file__} {mode}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, 3)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        kwargs = {
            'MPIsweeper': bool(int(sys.argv[1])),
            'MPIcontroller': bool(int(sys.argv[2])),
        }
        single_test(**kwargs)
    else:
        single_test()
