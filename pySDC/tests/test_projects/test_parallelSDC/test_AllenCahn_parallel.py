import pytest


@pytest.mark.mpi4py
def test_main():
    from pySDC.projects.parallelSDC.AllenCahn_parallel import main

    # try to import MPI here, will fail if things go wrong (and not later on in the subprocess part)
    import mpi4py

    main()
