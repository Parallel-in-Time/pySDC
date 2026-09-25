"""The transpose: moving a distributed array between pencil alignments with NCCL."""

import numpy as np
from mpi4py_fft.pencil import Pencil, Transfer


def get_slice(subtype):
    """Which part of the array an MPI subarray datatype describes.

    ``Transfer`` builds one datatype per peer to tell MPI which block to send. NCCL has no notion
    of datatypes and wants a contiguous buffer, so the block is read back out of the datatype and
    used to slice the array instead. Decoding what was just encoded is roundabout, but it keeps
    the blocks in one place: ``Transfer.__init__`` stays the single definition of who sends what.
    """
    _, _, info = subtype.decode()
    return tuple(slice(start, start + size) for start, size in zip(info['starts'], info['subsizes']))


#: What to tell NCCL a buffer of this dtype contains. Complex numbers go as pairs of reals, which
#: is what ``count_modifier`` below doubles the count for. Half precision is here because the
#: finite-difference problems already take a ``dtype``, and a field stored at ``float16`` has to
#: survive a transpose even where no transform accepts it.
def _nccl_dtype(dtype):
    from cupy.cuda import nccl

    table = {
        np.dtype('float16'): (nccl.NCCL_FLOAT16, 1),
        np.dtype('float32'): (nccl.NCCL_FLOAT32, 1),
        np.dtype('float64'): (nccl.NCCL_FLOAT64, 1),
        np.dtype('complex64'): (nccl.NCCL_FLOAT32, 2),
        np.dtype('complex128'): (nccl.NCCL_FLOAT64, 2),
        np.dtype('int32'): (nccl.NCCL_INT32, 1),
        np.dtype('int64'): (nccl.NCCL_INT64, 1),
    }
    if np.dtype(dtype) not in table:
        raise NotImplementedError(
            f'No NCCL type for {np.dtype(dtype)}. Add it to `_nccl_dtype` if the hardware has one; '
            'NCCL carries the bytes and does not care what they mean, so real and complex of the '
            'same width share an entry.'
        )
    return table[np.dtype(dtype)]


class NCCLTransfer(Transfer):
    """A :class:`mpi4py_fft.pencil.Transfer` that moves the data with NCCL.

    ``Transfer`` works out which block goes to which peer and then hands MPI a device pointer,
    which only works if MPI was built CUDA-aware. NCCL takes device pointers by construction and
    keeps the data on the device either way, so the block bookkeeping is inherited and only the
    exchange is replaced.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One communicator per transfer, torn down in `destroy` with the rest of it.
        #
        # Sharing them would be cheaper -- creating one is a collective with a broadcast in it, and
        # `redistribute` builds a transfer per call -- but a cache has to be keyed on the MPI
        # communicator, and `MPI_Comm_free` lets the handle be reused for an unrelated
        # communicator afterwards. That hands out an NCCL communicator built for a group that no
        # longer exists, which shows up as NCCL_ERROR_UNHANDLED_CUDA_ERROR in whichever test runs
        # next. `NCCLComm` caches for exactly this reason and never frees, which is issue #712;
        # doing it here would inherit the hazard without inheriting the reason.
        from cupy.cuda import nccl

        unique_id = self.comm.bcast(nccl.get_unique_id(), root=0)
        self.comm_nccl = nccl.NcclCommunicator(self.comm.size, unique_id, self.comm.rank)
        self.NCCL_dtype, self.count_modifier = _nccl_dtype(self.dtype)

    def _exchange(self, source, source_types, target, target_types):
        """Send every block of `source` to its peer and receive the peers' blocks into `target`."""
        import cupy as cp

        rank, size = self.comm.rank, self.comm.size
        stream = cp.cuda.get_current_stream()

        # A list rather than a dict keyed on the slice: two peers can want the same block, and a
        # dict would silently keep only the last of them.
        received = []

        # One group, so the sends and receives overlap rather than serialising. Offsetting the peer
        # by the rank stops every rank talking to rank 0 first.
        cp.cuda.nccl.groupStart()
        for step in range(size):
            send_to = (rank + step) % size
            recv_from = (rank - step + size) % size

            target_slice = get_slice(target_types[recv_from])
            # `cp.empty`, not `empty_like`: the blocks are views into a strided array, and
            # `empty_like` would copy those strides, while NCCL reads a flat buffer from the
            # pointer it is given. `ascontiguousarray` on the send side is a copy for the same
            # reason -- it is already contiguous only when the block happens to be.
            recv_buffer = cp.empty(target[target_slice].shape, dtype=target.dtype)
            send_buffer = cp.ascontiguousarray(source[get_slice(source_types[send_to])])

            self.comm_nccl.recv(
                recv_buffer.data.ptr, recv_buffer.size * self.count_modifier, self.NCCL_dtype, recv_from, stream.ptr
            )
            self.comm_nccl.send(
                send_buffer.data.ptr, send_buffer.size * self.count_modifier, self.NCCL_dtype, send_to, stream.ptr
            )
            received.append((target_slice, recv_buffer))
        cp.cuda.nccl.groupEnd()

        for target_slice, buffer in received:
            cp.copyto(target[target_slice], buffer)

    def forward(self, arrayA, arrayB):
        """Redistribute arrayA into arrayB."""
        self._exchange(arrayA, self._subtypesA, arrayB, self._subtypesB)

    def backward(self, arrayB, arrayA):
        """Redistribute arrayB into arrayA."""
        self._exchange(arrayB, self._subtypesB, arrayA, self._subtypesA)

    def destroy(self):
        """Free the MPI datatypes and the NCCL communicator together."""
        self.comm_nccl.destroy()
        super().destroy()


class CuPyPencil(Pencil):
    """A :class:`mpi4py_fft.pencil.Pencil` whose transfers go through NCCL.

    The decomposition itself is shapes, axes and start indices with no array in sight, so all of it
    is inherited; only the choice of how to move the data is ours.
    """

    def pencil(self, axis):
        """Keep the subclass when asked for the pencil aligned in another axis."""
        aligned = super().pencil(axis)
        return CuPyPencil(aligned.subcomm, aligned.shape, aligned.axis)

    def transfer(self, pencil, dtype):
        """The same checks and arguments as the base class, with an NCCL transfer at the end."""
        penA, penB = self, pencil
        assert penA.shape == penB.shape
        assert penA.axis != penB.axis
        for i in range(len(penA.shape)):
            if i not in (penA.axis, penB.axis):
                assert penA.subcomm[i] == penB.subcomm[i]
                assert penA.subshape[i] == penB.subshape[i]
        assert penA.subcomm[penB.axis] == penB.subcomm[penA.axis]

        axis = penB.axis
        shape = list(penA.subshape)
        shape[axis] = penA.shape[axis]
        return NCCLTransfer(penA.subcomm[axis], shape, dtype, penA.subshape, penA.axis, penB.subshape, penB.axis)
