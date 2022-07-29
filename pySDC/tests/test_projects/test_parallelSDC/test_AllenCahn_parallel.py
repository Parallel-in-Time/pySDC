from pySDC.projects.parallelSDC.AllenCahn_parallel import main


def test_main():

    # try to import MPI here, will fail if things go wrong (and not later on in the subprocess part)
    import mpi4py

    main()
