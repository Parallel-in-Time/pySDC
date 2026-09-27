from mpi4py import MPI
import firedrake as fd
import numpy as np


class _Requests:
    """
    The requests Firedrake returns for sending or receiving a function, one per subfunction, to be waited on together
    like a single request.
    """

    def __init__(self, requests):
        self.requests = requests

    def Wait(self):
        MPI.Request.Waitall(self.requests)

    wait = Wait


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

    _owns_comm_wold = False  # only communicators made by `Split` are freed by `Free`

    def Split(self, *args, **kwargs):
        """
        Split the world communicator with mpi4py's `Split` and return a new `FiredrakeEnsembleCommunicator` on the
        result, with the same spatial size.
        """
        split = FiredrakeEnsembleCommunicator(self.comm_wold.Split(*args, **kwargs), space_size=self.space_comm.size)
        split._owns_comm_wold = True
        return split

    @property
    def space_comm(self):
        """The spatial communicator of the ensemble, `ensemble.comm`."""
        return self.ensemble.comm

    @property
    def time_comm(self):
        """The time communicator of the ensemble, `ensemble.ensemble_comm`."""
        return self.ensemble.ensemble_comm

    def __getattr__(self, name):
        return getattr(self.time_comm, name)

    def Reduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        """
        Wrap `Reduce` on the time communicator for numpy arrays and `Ensemble.reduce` for Firedrake functions, which
        only takes `op=MPI.SUM` here.
        """
        if type(sendbuf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Reduce(sendbuf, recvbuf, op, root)
        else:
            assert op == MPI.SUM
            self.ensemble.reduce(sendbuf, recvbuf, root=root)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        """
        Wrap `Allreduce` on the time communicator for numpy arrays and `Ensemble.allreduce` for Firedrake functions,
        which only takes `op=MPI.SUM` here.
        """
        if type(sendbuf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Allreduce(sendbuf, recvbuf, op)
        else:
            assert op == MPI.SUM
            self.ensemble.allreduce(sendbuf, recvbuf)

    def Bcast(self, buf, root=0):
        """Wrap `Bcast` on the time communicator for numpy arrays and `Ensemble.bcast` for Firedrake functions."""
        if type(buf) in [np.ndarray]:
            self.ensemble.ensemble_comm.Bcast(buf, root)
        else:
            self.ensemble.bcast(buf, root=root)

    def Irecv(self, buf, source, tag=MPI.ANY_TAG):
        """
        Wrap `Irecv` on the time communicator for numpy arrays and lists, and `Ensemble.irecv` for Firedrake functions,
        whose requests, one per subfunction, are returned to be waited on together.
        """
        if type(buf) in [np.ndarray, list]:
            return self.ensemble.ensemble_comm.Irecv(buf=buf, source=source, tag=tag)
        return _Requests(self.ensemble.irecv(buf, source, tag=tag))

    def Isend(self, buf, dest, tag=0):
        """
        Wrap `Isend` on the time communicator for numpy arrays and lists, and `Ensemble.isend` for Firedrake functions,
        whose requests, one per subfunction, are returned to be waited on together.
        """
        if type(buf) in [np.ndarray, list]:
            return self.ensemble.ensemble_comm.Isend(buf=buf, dest=dest, tag=tag)
        return _Requests(self.ensemble.isend(buf, dest, tag=tag))

    def Free(self):
        """
        Free the world communicator if `Split` made it. The ensemble frees its own spatial and time communicators when
        it is garbage collected.
        """
        if self._owns_comm_wold:
            self.comm_wold.Free()


def get_ensemble(comm, space_size):
    """
    Create a Firedrake ensemble.

    Args:
        comm (MPI.Intracomm): MPI communicator, which will be split into time and space communicators
        space_size (int): Size of the spatial communicators

    Returns:
        firedrake.Ensemble: The ensemble
    """
    return fd.Ensemble(comm, space_size)
