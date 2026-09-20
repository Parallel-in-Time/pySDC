import pytest


@pytest.mark.mpi4py
@pytest.mark.parallel([2, 3])
def testOrder():
    """
    Test that the MPI sweepers reach the expected order.
    """
    from mpi4py import MPI
    from pySDC.projects.DAE.run.accuracy_check_MPI import check_order

    check_order(MPI.COMM_WORLD)


@pytest.mark.mpi4py
@pytest.mark.parallel(2)
@pytest.mark.parametrize("residual_type", ['full_abs', 'last_abs', 'full_rel', 'last_rel'])
@pytest.mark.parametrize("semi_implicit", [True, False])
@pytest.mark.parametrize("index_case", [1, 2])
@pytest.mark.parametrize("initial_guess", ['spread', 'zero', 'something_else'])
def testVersions(residual_type, semi_implicit, index_case, initial_guess):
    r"""
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes to use.
    residual_type : str
        Type of residual computation.
    semi_implicit : bool
        If True, semi-implicit sweeper is used.
    index_case : int
        Case of DAE index. Choose either between :math:`1` or :math:`2`.
    initial_guess : str
        Type of initial guess for simulation.
    """

    from mpi4py import MPI

    import numpy as np
    from pySDC.projects.DAE.run.accuracy_check_MPI import run
    from pySDC.core.errors import ParameterError

    semi_implicit = False if semi_implicit == 'False' else True
    num_nodes = MPI.COMM_WORLD.size

    dt = 0.1

    if initial_guess == 'something_else':
        with pytest.raises(ParameterError):
            _, _, _ = run(
                dt=dt,
                num_nodes=int(num_nodes),
                use_MPI=True,
                semi_implicit=semi_implicit,
                residual_type=residual_type,
                index_case=int(index_case),
                initial_guess=initial_guess,
            )

    else:
        MPI_uend, MPI_residual, _ = run(
            dt=dt,
            num_nodes=int(num_nodes),
            use_MPI=True,
            semi_implicit=semi_implicit,
            residual_type=residual_type,
            index_case=int(index_case),
            initial_guess=initial_guess,
        )

        nonMPI_uend, nonMPI_residual, _ = run(
            dt=dt,
            num_nodes=int(num_nodes),
            use_MPI=False,
            semi_implicit=semi_implicit,
            residual_type=residual_type,
            index_case=int(index_case),
            initial_guess=initial_guess,
        )

        assert np.allclose(MPI_uend, nonMPI_uend, atol=1e-14), 'Got different solutions at end point!'
        assert np.allclose(MPI_residual, nonMPI_residual, atol=1e-14), 'Got different residuals!'
