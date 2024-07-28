import pytest


def run_VdP(nNodes, nSteps, quad_type, useMPI=True):
    r"""
    Executes a run to solve numerically the Van der Pol with tests to check the `LogSolution` hook class.

    Parameters
    ----------
    nNodes : int or str
        Number of collocation nodes.
    nSteps : int or str
        Number of time steps to be executed.
    quad_type : str
        Type of quadrature.
    useMPI : bool, optional
        Either use MPI or not. By default ``True``.
    """

    import numpy as np
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted

    # initialize level parameters
    dt = 1e-1
    level_params = {
        'dt': dt,
        'restol': -1,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'QI': 'LU',
    }

    problem_params = {
        'newton_tol': 1e-12,
    }

    # initialize step parameters
    step_params = {'maxiter': 6}

    # initialize controller parameters
    controller_params = {'logger_level': 30, 'hook_class': [LogSolution]}

    if useMPI:
        from mpi4py import MPI
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper

        comm = MPI.COMM_WORLD
        sweeper_params.update({'comm': comm, 'num_nodes': comm.Get_size()})
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper

        nNodes = int(nNodes) if isinstance(nNodes, str) else nNodes
        sweeper_params.update({'num_nodes': nNodes})

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': vanderpol,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    lvl = controller.MS[0].levels[0]
    prob = controller.MS[0].levels[0].prob

    nSteps = int(nSteps) if isinstance(nSteps, str) else nSteps
    t0 = 0.0
    Tend = nSteps * dt

    uinit = prob.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    if useMPI:
        rank = comm.Get_rank()
        if rank == 0:
            # test for "u" and "u_dense"
            u = [me[1] for me in get_sorted(stats, type="u", sortby="time")]
            u_dense = [me[1] for me in get_sorted(stats, type="u_dense", sortby="time")][-1]

            assert np.allclose(uend, u[-1], atol=1e-14), "uend and stored u from hook does not match!"
            assert np.allclose(uend, u_dense[-1], atol=1e-14), "uend and stored u_dense at last node does not match"

            # test for "nodes_dense" - node that for 'RADAU-RIGHT' L.time is also stored in nodes!
            nodes_dense = [me[1] for me in get_sorted(stats, type="nodes", sortby="time")][-1]

            nodes_sweep = [lvl.time + lvl.dt * lvl.sweep.coll.nodes[m] for m in range(len(lvl.sweep.coll.nodes))]
            nodes_sweep = np.append([lvl.time], nodes_sweep) if not lvl.sweep.coll.left_is_node else nodes_sweep

            for m in range(len(nodes_sweep)):
                assert np.isclose(
                    nodes_dense[m], nodes_sweep[m], atol=1e-14
                ), "Nodes from sweeper does not match with stored nodes!"

            assert len(nodes_dense) == len(
                u_dense
            ), "Number of nodes does not match with number of u-values at these nodes!"

    else:
        # test for "u" and "u_dense"
        u = [me[1] for me in get_sorted(stats, type="u", sortby="time")]
        u_dense = [me[1] for me in get_sorted(stats, type="u_dense", sortby="time")][-1]

        assert np.allclose(uend, u[-1], atol=1e-14), "uend and stored u from hook does not match!"
        assert np.allclose(uend, u_dense[-1], atol=1e-14), "uend and stored u_dense at last node does not match"

        # test for "nodes_dense" - node that for 'RADAU-RIGHT' L.time is also stored in nodes!
        nodes_dense = [me[1] for me in get_sorted(stats, type="nodes", sortby="time")][-1]

        nodes_sweep = [lvl.time + lvl.dt * lvl.sweep.coll.nodes[m] for m in range(len(lvl.sweep.coll.nodes))]
        nodes_sweep = np.append([lvl.time], nodes_sweep) if not lvl.sweep.coll.left_is_node else nodes_sweep

        for m in range(len(nodes_sweep)):
            assert np.isclose(
                nodes_dense[m], nodes_sweep[m], atol=1e-14
            ), "Nodes from sweeper does not match with stored nodes!"

        assert len(nodes_dense) == len(
            u_dense
        ), "Number of nodes does not match with number of u-values at these nodes!"


@pytest.mark.base
@pytest.mark.parametrize("nNodes", [2, 3])
@pytest.mark.parametrize("nSteps", [2, 4, 6, 8])
@pytest.mark.parametrize("quad_type", ['RADAU-RIGHT', 'LOBATTO'])
def test_log_solution_nonMPI(nNodes, nSteps, quad_type, useMPI=False):
    """Non-MPI test."""
    run_VdP(nNodes=nNodes, nSteps=nSteps, quad_type=quad_type, useMPI=useMPI)


@pytest.mark.mpi4py
@pytest.mark.parametrize("nNodes", [2, 3])
@pytest.mark.parametrize("nSteps", [2, 4, 6, 8])
@pytest.mark.parametrize("quad_type", ['RADAU-RIGHT', 'LOBATTO'])
def test_log_solution_MPI(nNodes, nSteps, quad_type):
    """MPI test."""

    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {nNodes} python {__file__} {nNodes} {nSteps} {quad_type}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        nNodes,
    )


if __name__ == '__main__':
    import sys

    run_VdP(sys.argv[1], sys.argv[2], sys.argv[3])
