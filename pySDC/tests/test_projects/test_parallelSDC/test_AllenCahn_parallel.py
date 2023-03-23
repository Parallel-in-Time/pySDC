import pytest


@pytest.mark.mpi4py
def test_main():
    from pySDC.projects.parallelSDC.AllenCahn_parallel import main

    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py

        del mpi4py
    except ImportError:
        raise ImportError('petsc tests need mpi4py')

    main()
