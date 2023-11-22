import pytest


@pytest.mark.mpi4py
def test_schroedinger_solution():
    from mpi4py import MPI
    import numpy as np
    from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
    from pySDC.helpers.stats_helper import get_sorted

    stats, _, _ = run_Schroedinger(space_comm=MPI.COMM_WORLD)
    k_mean = np.mean([me[1] for me in get_sorted(stats, type='k')])
    assert (
        k_mean < 17
    ), f"Got too many iterations in Schroedinger test! Expected less then 17 on average, but got {k_mean:.2f}!"


@pytest.mark.mpi4py
def test_schroedinger_fault_insertion():
    from mpi4py import MPI
    import numpy as np
    from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
    from pySDC.projects.Resilience.fault_injection import FaultInjector
    from pySDC.helpers.stats_helper import get_sorted

    fault_stuff = FaultInjector.generate_fault_stuff_single_fault(
        bit=0,
        iteration=5,
        problem_pos=[20, 30],
        level_number=0,
        node=3,
        time=0.1,
        rank=0,
    )

    stats, _, _ = run_Schroedinger(space_comm=MPI.COMM_WORLD, fault_stuff=fault_stuff)
    k_mean = np.mean([me[1] for me in get_sorted(stats, type='k')])
    assert (
        k_mean > 17
    ), f"Got too few iterations in Schroedinger test! Expected more then 17 on average because we need to fix the fault, but got {k_mean:.2f}!"
