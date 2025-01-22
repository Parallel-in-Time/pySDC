from mpi4py import MPI
import firedrake as fd
import numpy as np


class FiredrakeEnsembleCommunicator:
    """
    Ensemble communicator for performing multiple similar distributed simulations with Firedrake, see https://www.firedrakeproject.org/firedrake/parallelism.html
    This is intended to do space-time parallelism in pySDC.
    This class wraps the time communicator. All requests that are not overloaded are passed to the time communicator. For instance, `ensemble.rank` will return the rank in the time communicator.
    Some operations are overloaded to use the interface of the MPI communicator but handles communication with the ensemble communicator instead.
    """

    def __init__(self, comm, space_size):
        """
        Args:
            comm (MPI.Intracomm): MPI communicator, which will be split into time and space communicators
            space_size (int): Size of the spatial communicators

        Attributes:
            ensemble (firedrake.Ensemble): Ensemble communicator
        """
        self.ensemble = fd.Ensemble(comm, space_size)

    @property
    def space_comm(self):
        return self.ensemble.comm

    @property
    def time_comm(self):
        return self.ensemble.ensemble_comm

    def __getattr__(self, name):
        return getattr(self.time_comm, name)

    def Reduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        if type(sendbuf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Reduce(sendbuf, recvbuf, op, root)
        else:
            assert op == MPI.SUM
            self.ensemble.reduce(sendbuf, recvbuf, root=root)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        if type(sendbuf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Allreduce(sendbuf, recvbuf, op)
        else:
            assert op == MPI.SUM
            self.ensemble.allreduce(sendbuf, recvbuf)

    def Bcast(self, buf, root=0):
        if type(buf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Bcast(buf, root)
        else:
            self.ensemble.bcast(buf, root=root)


def get_ensemble(comm, space_size):
    return fd.Ensemble(comm, space_size)
