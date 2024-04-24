import numpy as np
import torch

from pySDC.core.Errors import DataError

try:
    # TODO : mpi4py cannot be imported before dolfin when using fenics mesh
    # see https://github.com/Parallel-in-Time/pySDC/pull/285#discussion_r1145850590
    # This should be dealt with at some point
    from mpi4py import MPI
except ImportError:
    MPI = None


class Tensor(torch.Tensor):
    """
    Wrapper for PyTorch tensor.
    Be aware that this is totally WIP! Should be fine to count iterations, but desperately needs cleaning up if this project goes much further!

    TODO: Have to update `torch/multiprocessing/reductions.py` in order to share this datatype across processes.

    Attributes:
        _comm: MPI communicator or None
    """

    @staticmethod
    def __new__(cls, init, val=0.0, *args, **kwargs):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, Tensor):
            obj = super().__new__(cls, init)
            obj[:] = init[:]
            obj._comm = init._comm
        elif (
            isinstance(init, tuple)
            # and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            # and isinstance(init[2], np.dtype)
        ):
            obj = super().__new__(cls, init[0].clone())
            obj.fill_(val)
            obj._comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    @property
    def comm(self):
        """
        Getter for the communicator
        """
        return self._comm

    def __array_finalize__(self, obj):
        """
        Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
        """
        if obj is None:
            return
        self._comm = getattr(obj, '_comm', None)

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(torch.amax(torch.abs(self)))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = 0.0
                global_absval = max(self.comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return float(global_absval)

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.Issend(self[:], dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        return comm.Irecv(self[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self[:], root=root)
        return self
