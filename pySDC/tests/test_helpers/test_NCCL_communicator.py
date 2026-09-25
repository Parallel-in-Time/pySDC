"""
Tests for `NCCLComm` itself, rather than for a sweeper that happens to use one.

Most of this class is a translation layer -- MPI datatypes and operations to their NCCL
equivalents, and device buffers to raw pointers -- and a wrong entry in one of those tables shows
up as a subtly wrong number somewhere in a sweep, if it shows up at all. Checking the translation
where it happens says what actually broke.
"""

import numpy as np
import pytest


@pytest.mark.cupy
def test_dtypes_translate_to_their_NCCL_equivalents():
    """NCCL has no complex numbers, so a complex array travels as twice as many reals."""
    import cupy as cp
    from cupy.cuda import nccl

    from pySDC.helpers.NCCL_communicator import NCCLComm

    equivalent = {
        'float32': nccl.NCCL_FLOAT32,
        'complex64': nccl.NCCL_FLOAT32,
        'float64': nccl.NCCL_FLOAT64,
        'complex128': nccl.NCCL_FLOAT64,
        'int32': nccl.NCCL_INT32,
        'int64': nccl.NCCL_INT64,
    }
    for dtype, expected in equivalent.items():
        assert NCCLComm.get_dtype(cp.empty(4, dtype=dtype)) == expected, f'wrong NCCL dtype for {dtype}'

    with pytest.raises(NotImplementedError):
        NCCLComm.get_dtype(cp.empty(4, dtype='float16'))


@pytest.mark.cupy
def test_complex_arrays_are_counted_as_twice_as_many_reals():
    """The other half of sending complex data as real: the count has to be doubled to match."""
    import cupy as cp

    from pySDC.helpers.NCCL_communicator import NCCLComm

    assert NCCLComm.get_count(cp.empty(8, dtype='float64')) == 8
    assert NCCLComm.get_count(cp.empty(8, dtype='complex128')) == 16


@pytest.mark.cupy
def test_operations_translate_to_their_NCCL_equivalents():
    from cupy.cuda import nccl
    from mpi4py import MPI

    from pySDC.helpers.NCCL_communicator import NCCLComm

    comm = NCCLComm(MPI.COMM_WORLD)

    # a list of pairs rather than a dict: `MPI.Op` is not hashable
    equivalent = [
        (MPI.SUM, nccl.NCCL_SUM),
        (MPI.PROD, nccl.NCCL_PROD),
        (MPI.MAX, nccl.NCCL_MAX),
        (MPI.MIN, nccl.NCCL_MIN),
    ]
    for MPI_op, expected in equivalent:
        assert comm.get_op(MPI_op) == expected, f'wrong NCCL operation for {MPI_op}'

    with pytest.raises(NotImplementedError):
        comm.get_op(MPI.LAND)


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_host_buffers_are_handed_back_to_MPI():
    """Only device buffers can go through NCCL; anything else has to reach MPI unchanged."""
    from mpi4py import MPI

    from pySDC.helpers.NCCL_communicator import NCCLComm

    comm = NCCLComm(MPI.COMM_WORLD)

    total = np.zeros(4)
    comm.Allreduce(np.ones(4), total, op=MPI.SUM)
    assert np.allclose(total, comm.size), f'host Allreduce gave {total}'

    total = np.zeros(4)
    comm.Reduce(np.ones(4), total, op=MPI.SUM, root=0)
    if comm.rank == 0:
        assert np.allclose(total, comm.size), f'host Reduce gave {total}'

    # every rank starts from its own value, so a broadcast that did nothing would be visible
    shared = np.full(4, float(comm.rank + 1))
    comm.Bcast(shared, root=0)
    assert np.allclose(shared, 1.0), f'host Bcast gave {shared}'

    comm.Barrier()


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_device_arrays_are_synchronised_before_MPI_reads_them():
    """`reduce` and `allreduce` pickle through MPI, which reads the data from the host.

    Whatever the device still has in flight has to have landed by then, which is why these two
    synchronise first when handed something that lives on the device.
    """
    import cupy as cp
    from mpi4py import MPI

    from pySDC.helpers.NCCL_communicator import NCCLComm

    comm = NCCLComm(MPI.COMM_WORLD)
    expected = sum(rank + 1 for rank in range(comm.size))

    # a fresh multiplication, so there is something outstanding on the stream to synchronise on
    on_device = cp.ones(4) * (comm.rank + 1)

    total = comm.allreduce(on_device, op=MPI.SUM)
    assert cp.allclose(cp.asarray(total), expected), f'device allreduce gave {total}'

    total = comm.reduce(on_device, op=MPI.SUM, root=0)
    if comm.rank == 0:
        assert cp.allclose(cp.asarray(total), expected), f'device reduce gave {total}'
