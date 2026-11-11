import pytest


@pytest.mark.mpi4py
@pytest.mark.parallel(4)
def test_every_rank_returns_the_final_solution():
    """
    Six steps on four ranks leave ranks 2 and 3 idle in the second block. They used to return the
    solution at the end of the first block instead of the one at Tend.
    """
    import numpy as np
    from mpi4py import MPI

    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    comm = MPI.COMM_WORLD
    description = {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': np.array([-1.0]), 'u0': 1.0},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QI': 'LU'},
        'level_params': {'dt': 0.1},
        'step_params': {'maxiter': 2},
    }
    controller = controller_MPI(controller_params={'logger_level': 30}, description=description, comm=comm)
    uend, _ = controller.run(controller.S.levels[0].prob.u_exact(0.0), 0.0, 0.6)

    # the step that ends last holds the final solution, and returning it is a copy, so exactly equal
    S = controller.S
    last = max(comm.allgather((S.time + S.dt, S.levels[0].uend)), key=lambda me: me[0])[1]
    assert np.array_equal(uend, last), f'rank {comm.rank} returned {uend}, but the run ended at {last}'
