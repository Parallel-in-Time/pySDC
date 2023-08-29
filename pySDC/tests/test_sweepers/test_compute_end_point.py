import pytest


def compute_end_point_generic_implicit(num_nodes, useMPI):
    import numpy as np
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    description = {}
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = {'num_nodes': num_nodes, 'quad_type': 'GAUSS', 'do_coll_update': True, 'comm': comm}
    description['problem_params'] = {'degree': num_nodes + 1}
    description['level_params'] = {'dt': 1.0}
    description['step_params'] = {}
    description['problem_class'] = polynomial_testequation

    controller = controller_nonMPI(1, {'logger_level': 30}, description)
    step = controller.MS[0]
    level = step.levels[0]
    prob = level.prob
    sweep = level.sweep
    nodes = np.append([0], level.sweep.coll.nodes)

    for i in range(len(level.u)):
        level.u[i] = prob.u_exact(nodes[i])
        level.f[i] = prob.eval_f(level.u[i], nodes[i])
    sweep.compute_end_point()

    error = abs(level.uend - prob.u_exact(1.0))
    assert error < 1e-15, f'Failed to compute end point! Error: {error}'
    print(f'Passed with error to exact end point: {error}')

    assert abs(level.uend - level.u[0]) > 1e-15, 'Solution is constant!'


@pytest.mark.base
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_compute_end_point_generic_implicit(num_nodes):
    compute_end_point_generic_implicit(num_nodes, False)


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_compute_end_point_generic_implicit_MPI(num_nodes):
    import subprocess
    import os

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_nodes} python {__file__} MPI {num_nodes}".split()

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
            'num_nodes': int(sys.argv[2]),
        }
    compute_end_point_generic_implicit(**kwargs)
