"""
Ownership of ``mpi4py-fft`` transform objects.

A ``PFFT`` allocates MPI communicators -- ``MPI_Cart_create`` plus an ``MPI_Cart_sub`` per axis --
and does not release them when the Python object is collected. A process that builds many of them
therefore runs out of communicator contexts: measured at **1022 creations** on MPICH 4.3.2, failing
with ``MPI_Cart_create(MPI_COMM_WORLD, ndims=3, ...) failed``.

This never surfaced while every MPI test got its own short-lived ``mpirun``, since no single process
lived long enough to reach the limit. It does surface once a whole test session shares one process,
and it would equally hit any long-running driver that builds many problems.

``PFFT.destroy()`` is the release path. It is idempotent, and arrays already created with
``newDistArray`` stay usable afterwards, so tying it to garbage collection is safe.
"""

from mpi4py_fft import PFFT as _PFFT


class PFFT(_PFFT):
    """A :class:`mpi4py_fft.PFFT` that frees its MPI communicators when it is collected."""

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            # Nothing useful can be done here: this runs during garbage collection, possibly at
            # interpreter shutdown with MPI already finalized, and raising would only print noise.
            pass
