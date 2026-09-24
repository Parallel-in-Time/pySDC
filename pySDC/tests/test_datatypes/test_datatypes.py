import pytest


def get_dtype(name):
    if name == 'Tensor':
        from pySDC.playgrounds.ML_initial_guess.tensor import Tensor as dtype_cls
    elif name in ['mesh', 'imex_mesh']:
        import pySDC.implementations.datatype_classes.mesh as mesh

        dtype_cls = eval(f'mesh.{name}')
    else:
        raise NotImplementedError(f'Don\'t know a dtype of name {name!r}!')

    return dtype_cls


def single_test(name, useMPI=False):
    """
    This test checks that the communicator and datatype are maintained when generating new instances.
    Also, it makes sure that you can supply different communicators.
    """
    import numpy as np

    dtype_cls = get_dtype(name)

    shape = (5,)
    comm = None
    dtype = np.dtype('f')

    if useMPI:
        from mpi4py import MPI

        comm_wd = MPI.COMM_WORLD
        comm = comm_wd.Split(comm_wd.rank < comm_wd.size - 1)

        expected_rank = comm_wd.rank % (comm_wd.size - 1)

    init = (shape, comm, dtype)

    a = dtype_cls(init, val=1.0)
    b = dtype_cls(init, val=99.0)
    c = dtype_cls(a)
    d = a + b

    for me in [a, b, c, d]:
        assert type(me) == dtype_cls
        assert me.comm == comm

        if hasattr(me, 'shape') and not hasattr(me, 'components'):
            assert me.shape == shape

        if useMPI:
            assert comm.rank == expected_rank
            assert comm.size < comm_wd.size


@pytest.mark.pytorch
@pytest.mark.parallel(4)
def test_PyTorch_dtype_MPI():
    single_test('Tensor', True)


@pytest.mark.pytorch
def test_PyTorch_dtype():
    single_test('Tensor', False)


@pytest.mark.mpi4py
@pytest.mark.parallel(4)
@pytest.mark.parametrize('name', ['mesh', 'imex_mesh'])
def test_mesh_dtypes_MPI(name):
    single_test(name, True)


@pytest.mark.base
@pytest.mark.parametrize('name', ['mesh', 'imex_mesh'])
def test_mesh_dtypes(name):
    single_test(name, False)


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_cupy_mesh_norm_is_taken_across_ranks():
    """`abs()` on a space-parallel mesh is a global maximum, and on the GPU it goes through NCCL.

    The CPU twin reduces a Python float through MPI; this one hands NCCL two device buffers
    instead, which is a different branch of `__abs__` and the only one a GPU ever takes.
    """
    import cupy as cp
    from mpi4py import MPI

    from pySDC.helpers.NCCL_communicator import NCCLComm
    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

    comm = NCCLComm(MPI.COMM_WORLD)

    # `__new__` keeps the communicator on the class rather than on the instance, so put back
    # whatever was there and leave the rest of the session as it was found
    previously = cupy_mesh.comm
    try:
        mesh = cupy_mesh(init=((4,), comm, cp.dtype('float64')))

        # a different local maximum on every rank, so a norm that forgot to reduce would come
        # back wrong on every rank but the last
        mesh[:] = comm.rank + 1

        assert abs(mesh) == comm.size, f'rank {comm.rank} did not get the global maximum'
    finally:
        cupy_mesh.comm = previously
