import pytest


def run(use_MPI, num_nodes, quad_type, residual_type, imex, init_guess, useNCCL, ML):
    """
    Run a single sweep for a problem and compute the solution at the end point with a sweeper as specified.

    Args:
        use_MPI (bool): Use the MPI version of the sweeper or not
        num_nodes (int): The number of nodes to use
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
        init_guess (str): which initial guess should be used
        useNCCL (bool): ...
        ML (int): Number of levels in space

    Returns:
        pySDC.Level.level: The level containing relevant data
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if not imex:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        if ML > 1:
            from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced as problem_class
        else:
            from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
    else:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced as problem_class

    dt = 1e-1
    description = {}
    sweeper_params = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'QI': 'IEpar',
        'QE': 'PIC',
        "initial_guess": init_guess,
    }
    problem_params = {}

    if useNCCL:
        from pySDC.helpers.NCCL_communicator import NCCLComm
        from mpi4py import MPI

        sweeper_params['comm'] = NCCLComm(MPI.COMM_WORLD)
        problem_params['useGPU'] = True

    if ML > 1:
        from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

        description['space_transfer_class'] = mesh_to_mesh

        problem_params['nvars'] = [2 ** (ML - i) for i in range(ML)]
        if use_MPI:
            from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI

            description['base_transfer_class'] = base_transfer_MPI

    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = {'dt': dt, 'residual_type': residual_type}
    description['step_params'] = {'maxiter': 1}

    controller = controller_nonMPI(1, {'logger_level': 30}, description)

    if imex:
        u0 = controller.MS[0].levels[0].prob.u_exact(0)
    else:
        u0 = controller.MS[0].levels[0].prob.u_exact(0) + 1.0
    controller.run(u0, 0, dt)
    controller.MS[0].levels[0].sweep.compute_end_point()
    return controller.MS[0].levels[0]


def individual_test(**kwargs):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.
    """
    if kwargs['useNCCL']:
        import cupy as xp
    else:
        import numpy as xp

    MPI = run(**kwargs, use_MPI=True)
    nonMPI = run(**kwargs, use_MPI=False)

    assert xp.allclose(
        MPI.uend, nonMPI.uend, rtol=0, atol=1e-14
    ), f'Got different solutions at end point! {MPI.uend=} {nonMPI.uend=}'
    assert xp.allclose(MPI.status.residual, nonMPI.status.residual, rtol=0, atol=1e-14), 'Got different residuals!'


@pytest.mark.mpi4py
@pytest.mark.parallel(2)
@pytest.mark.parametrize("quad_type", ['GAUSS', 'RADAU-RIGHT'])
@pytest.mark.parametrize("residual_type", ['last_abs', 'full_rel'])
@pytest.mark.parametrize("imex", [True, False])
@pytest.mark.parametrize("init_guess", ['spread', 'copy', 'zero'])
@pytest.mark.parametrize("ML", [1, 2, 3])
def test_sweeper(quad_type, residual_type, imex, init_guess, ML):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Args:
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
    """
    from mpi4py import MPI

    individual_test(
        num_nodes=MPI.COMM_WORLD.size,
        quad_type=quad_type,
        residual_type=residual_type,
        imex=imex,
        init_guess=init_guess,
        useNCCL=False,
        ML=ML,
    )


@pytest.mark.cupy
@pytest.mark.parallel(2)
@pytest.mark.parametrize("quad_type", ['GAUSS', 'RADAU-RIGHT'])
@pytest.mark.parametrize("residual_type", ['last_abs', 'full_rel'])
@pytest.mark.parametrize("imex", [False])
@pytest.mark.parametrize("init_guess", ['spread', 'copy', 'zero'])
@pytest.mark.parametrize("ML", [1, 2, 3])
def test_sweeper_NCCL(quad_type, residual_type, imex, init_guess, ML):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Args:
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
    """
    from mpi4py import MPI

    individual_test(
        num_nodes=MPI.COMM_WORLD.size,
        quad_type=quad_type,
        residual_type=residual_type,
        imex=imex,
        init_guess=init_guess,
        useNCCL=True,
        ML=ML,
    )


def run_with_distributed_space(space_comm, node_comm, nvars=(32, 32), dt=1e-2):
    """One step of a GPU run, with the nodes over `node_comm` and the space over `space_comm`.

    `node_comm` of None runs every node on this rank with the ordinary sweeper, which is the
    reference the parallel versions have to reproduce.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT

    if node_comm is None:
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        sweeper_params = {'num_nodes': 2}
    else:
        from pySDC.helpers.NCCL_communicator import NCCLComm
        from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class

        sweeper_params = {'num_nodes': node_comm.size, 'comm': NCCLComm(node_comm)}

    # The MPI sweeper needs a diagonal preconditioner to have anything to parallelise, which rules
    # out `LU`; `MIN-SR-S` is the diagonal one worth running. The serial reference uses the same,
    # so the comparison is of the parallelisation and not of the preconditioner.
    sweeper_params.update({'quad_type': 'RADAU-RIGHT', 'QI': 'MIN-SR-S', 'QE': 'PIC'})

    description = {
        'problem_class': IMEX_Laplacian_MPIFFT,
        'problem_params': {'nvars': nvars, 'comm': space_comm, 'useGPU': True, 'spectral': False},
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'dt': dt},
        'step_params': {'maxiter': 3},
    }

    controller = controller_nonMPI(1, {'logger_level': 30}, description)
    level = controller.MS[0].levels[0]
    prob = level.prob

    u0 = prob.u_init
    u0[...] = prob.xp.sin(prob.X[0]) * prob.xp.sin(prob.X[1])

    controller.run(u0, 0, dt)
    level.sweep.compute_end_point()
    return level


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_node_parallel_on_GPU_serial_space():
    """Two collocation nodes on two GPUs, each holding the whole spatial domain.

    `test_sweeper_NCCL` above already spreads nodes over GPUs, but on a finite-difference problem
    that has no space communicator at all. This is the same split on a problem that does, which is
    what makes it comparable with the space-distributed version below.
    """
    from mpi4py import MPI

    parallel = run_with_distributed_space(MPI.COMM_SELF, MPI.COMM_WORLD)
    serial = run_with_distributed_space(MPI.COMM_SELF, None)

    import cupy as cp

    assert cp.allclose(parallel.uend, serial.uend, rtol=0, atol=1e-13), 'node-parallel run differs from the serial one'


@pytest.mark.cupy
@pytest.mark.parallel(4)
def test_node_parallel_on_GPU_distributed_space():
    """Two collocation nodes by two ranks in space, which nothing else covers.

    The nodes talk over NCCL while the spatial transforms redistribute over a communicator of their
    own, so the two decompositions have to stay out of each other's way. Checked against a run with
    neither, on the slice of the global array this rank holds.
    """
    import cupy as cp
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    assert world.size == 4, f'this test decomposes four ranks as 2x2, not {world.size}'

    # rank r sits at node r // 2 and space r % 2, so ranks sharing a node are together in space
    space_comm = world.Split(color=world.rank // 2)
    node_comm = world.Split(color=world.rank % 2)

    try:
        parallel = run_with_distributed_space(space_comm, node_comm)
        serial = run_with_distributed_space(MPI.COMM_SELF, None)

        mine = serial.uend[parallel.prob.fft.local_slice(False)]
        assert cp.allclose(parallel.uend, mine, rtol=0, atol=1e-13), 'the 2x2 run differs from the serial one'

        # and both decompositions have to have done something, or the comparison is vacuous
        assert parallel.prob.fft.shape(False) != parallel.prob.fft.global_shape(), 'space was not distributed'
        assert parallel.sweep.comm.size == 2, 'the nodes were not distributed'
    finally:
        space_comm.Free()
        node_comm.Free()
