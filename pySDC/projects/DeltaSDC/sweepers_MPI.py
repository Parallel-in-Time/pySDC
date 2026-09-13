r"""
Node-parallel reduced precision, emulated, on top of pySDC's node-parallel delta-form hierarchy.

The hierarchy is in :mod:`pySDC.implementations.sweeper_classes.delta_form_MPI` and
:mod:`pySDC.implementations.transfer_classes.BaseTransferDeltaMPI`. What is here is the emulation
layered on it, the same way :mod:`.mlsdc` layers it on the serial hierarchy.
"""

from mpi4py import MPI

from pySDC.implementations.sweeper_classes.delta_form_MPI import delta_implicit_MPI as _delta_implicit_MPI
from pySDC.implementations.transfer_classes.BaseTransferDeltaMPI import delta_transfer_MPI as _delta_transfer_MPI
from pySDC.projects.DeltaSDC.cascade import PrecisionCascade
from pySDC.projects.DeltaSDC.mlsdc import RoundedLevelMixin, round_level


class delta_implicit_MPI_rounded(RoundedLevelMixin, _delta_implicit_MPI):
    """Node-parallel sweeper whose level storage is rounded through ``level_precision``."""


class delta_transfer_MPI(_delta_transfer_MPI):
    """Node-parallel delta-form transfer that rounds each level's storage after touching it."""

    def restrict(self):
        """
        Restrict, then round the coarse level.

        Returns
        -------
        None
        """
        super().restrict()
        round_level(self.coarse)
        self.coarse.u0_reference = self.coarse.prob.dtype_u(self.coarse.u[0])
        return None

    def prolong(self):
        """
        Prolong, then round the fine level.

        Returns
        -------
        None
        """
        super().prolong()
        round_level(self.fine)
        return None


class delta_implicit_MPI_cascade(PrecisionCascade, delta_implicit_MPI_rounded):
    """
    Node-parallel counterpart of :class:`.cascade.delta_implicit_cascade`.

    The indicator has to be reduced across the node communicator. Each rank holds one collocation
    node and so sees only part of both norms, and a rank that stepped while its neighbours did not
    would leave the level stored at two different precisions at once.
    """

    def cascade_norms(self):
        """
        The indicator's two norms, reduced over the node communicator.

        Returns
        -------
        tuple
            ``(|delta|, |u|)`` in the infinity norm, agreed by every rank.
        """
        delta, state = super().cascade_norms()
        return self.comm.allreduce(delta, op=MPI.MAX), self.comm.allreduce(state, op=MPI.MAX)
