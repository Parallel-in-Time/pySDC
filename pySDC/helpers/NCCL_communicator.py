from mpi4py import MPI
from cupy.cuda import nccl
import cupy as cp
import numpy as np


class NCCLComm(object):
    """
    Wraps an MPI communicator and performs some calls to NCCL functions instead.
    """

    def __init__(self, comm):
        """
        Args:
            comm (mpi4py.Intracomm): MPI communicator
        """
        self.commMPI = comm

        uid = comm.bcast(nccl.get_unique_id(), root=0)
        self.commNCCL = nccl.NcclCommunicator(comm.size, uid, comm.rank)

    def __getattr__(self, name):
        """
        Pass calls that are not explicitly overridden by NCCL functionality on to the MPI communicator.
        When performing any operations that depend on data, we have to synchronize host and device beforehand.

        Args:
            Name (str): Name of the requested attribute
        """
        if name not in ['size', 'rank', 'Get_rank', 'Get_size', 'Split', 'Create_cart', 'Is_inter', 'Get_topology']:
            cp.cuda.get_current_stream().synchronize()

        return getattr(self.commMPI, name)

    @staticmethod
    def get_dtype(data):
        """
        As NCCL doesn't support complex numbers, we have to act as if we're sending two real numbers if using complex.
        """
        dtype = data.dtype
        if dtype in [np.dtype('float32'), np.dtype('complex64')]:
            return nccl.NCCL_FLOAT32
        elif dtype in [np.dtype('float64'), np.dtype('complex128')]:
            return nccl.NCCL_FLOAT64
        elif dtype in [np.dtype('int32')]:
            return nccl.NCCL_INT32
        elif dtype in [np.dtype('int64')]:
            return nccl.NCCL_INT64
        else:
            raise NotImplementedError(f'Don\'t know what NCCL dtype to use to send data of dtype {data.dtype}!')

    @staticmethod
    def get_count(data):
        """
        As NCCL doesn't support complex numbers, we have to act as if we're sending two real numbers if using complex.
        """
        if cp.iscomplexobj(data):
            return data.size * 2
        else:
            return data.size

    def get_op(self, MPI_op):
        if MPI_op == MPI.SUM:
            return nccl.NCCL_SUM
        elif MPI_op == MPI.PROD:
            return nccl.NCCL_PROD
        elif MPI_op == MPI.MAX:
            return nccl.NCCL_MAX
        elif MPI_op == MPI.MIN:
            return nccl.NCCL_MIN
        else:
            raise NotImplementedError('Don\'t know what NCCL operation to use to replace this MPI operation!')

    def reduce(self, sendobj, op=MPI.SUM, root=0):
        sync = False
        if hasattr(sendobj, 'data'):
            if hasattr(sendobj.data, 'ptr'):
                sync = True
        if sync:
            cp.cuda.Device().synchronize()

        return self.commMPI.reduce(sendobj, op=op, root=root)

    def allreduce(self, sendobj, op=MPI.SUM):
        sync = False
        if hasattr(sendobj, 'data'):
            if hasattr(sendobj.data, 'ptr'):
                sync = True
        if sync:
            cp.cuda.Device().synchronize()

        return self.commMPI.allreduce(sendobj, op=op)

    def Reduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        if not hasattr(sendbuf.data, 'ptr'):
            return self.commMPI.Reduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op, root=root)

        dtype = self.get_dtype(sendbuf)
        count = self.get_count(sendbuf)
        op = self.get_op(op)
        recvbuf = cp.empty(1) if recvbuf is None else recvbuf
        stream = cp.cuda.get_current_stream()

        self.commNCCL.reduce(
            sendbuf=sendbuf.data.ptr,
            recvbuf=recvbuf.data.ptr,
            count=count,
            datatype=dtype,
            op=op,
            root=root,
            stream=stream.ptr,
        )

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        if not hasattr(sendbuf.data, 'ptr'):
            return self.commMPI.Allreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op)

        dtype = self.get_dtype(sendbuf)
        count = self.get_count(sendbuf)
        op = self.get_op(op)
        stream = cp.cuda.get_current_stream()

        self.commNCCL.allReduce(
            sendbuf=sendbuf.data.ptr, recvbuf=recvbuf.data.ptr, count=count, datatype=dtype, op=op, stream=stream.ptr
        )

    def Bcast(self, buf, root=0):
        if not hasattr(buf.data, 'ptr'):
            return self.commMPI.Bcast(buf=buf, root=root)

        dtype = self.get_dtype(buf)
        count = self.get_count(buf)
        stream = cp.cuda.get_current_stream()

        self.commNCCL.bcast(buff=buf.data.ptr, count=count, datatype=dtype, root=root, stream=stream.ptr)

    def Barrier(self):
        cp.cuda.get_current_stream().synchronize()
        self.commMPI.Barrier()
