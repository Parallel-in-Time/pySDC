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
        self.comm_wold = comm

    def Split(self, *args, **kwargs):
        return FiredrakeEnsembleCommunicator(self.comm_wold.Split(*args, **kwargs), space_size=self.space_comm.size)

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

    def Irecv(self, buf, source, tag=MPI.ANY_TAG):
        if type(buf) in [np.ndarray, list]:
            return self.ensemble.ensemble_comm.Irecv(buf=buf, source=source, tag=tag)
        return self.ensemble.irecv(buf, source, tag=tag)[0]

    def Isend(self, buf, dest, tag=MPI.ANY_TAG):
        if type(buf) in [np.ndarray, list]:
            return self.ensemble.ensemble_comm.Isend(buf=buf, dest=dest, tag=tag)
        return self.ensemble.isend(buf, dest, tag=tag)[0]

    def Free(self):
        del self


def get_ensemble(comm, space_size):
    return fd.Ensemble(comm, space_size)
