import pytest


@pytest.mark.mpi4py
def test_schroedinger():
    from mpi4py import MPI
    import numpy as np
    from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
    from pySDC.helpers.stats_helper import get_sorted

    stats, _, _ = run_Schroedinger(space_comm=MPI.COMM_WORLD)
    k_mean = np.mean([me[1] for me in get_sorted(stats, type='k')])
    assert (
        k_mean < 17
    ), f"Got too many iterations in Schroedinger test! Expected less then 17 on average, but got {k_mean:.2f}!"
