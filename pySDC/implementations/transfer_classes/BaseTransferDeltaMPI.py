r"""
Node-parallel counterpart of :mod:`pySDC.implementations.transfer_classes.BaseTransferDelta`.

Same two identities -- the coarse residual is the restricted fine residual, and the coarse-grid
correction is the sum of the sweep's own increments -- with every loop over collocation nodes
written as the reduction the node-parallel layout needs, since one rank holds one node.

One difference from the serial transfer: this one still builds the FAS :math:`\tau`, which the
delta-form hierarchy then never reads. Skipping it is the saving
:meth:`BaseTransferDelta.delta_transfer.restrict_state` takes, and it has not been written for the
node-parallel layout yet.
"""

from mpi4py import MPI

from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI


class delta_transfer_MPI(base_transfer_MPI):
    """
    Node-parallel counterpart of
    :class:`~pySDC.implementations.transfer_classes.BaseTransferDelta.delta_transfer`.

    Same two identities -- the coarse residual is the restricted fine residual, and the coarse-grid
    correction is the sum of the sweep's own increments -- with the collocation transfer written as
    a reduction, since each rank holds one node.
    """

    def restrict(self):
        """
        Restrict as usual, then hand the coarse level the restricted fine residual.

        Returns
        -------
        None
        """
        SF, SG = self.fine.sweep, self.coarse.sweep
        SF._delta_setup()
        SG._delta_setup()
        eps_F = SF._residual_nodes()[0]
        super().restrict()

        CF, CG, PG = self.comm_fine, self.comm_coarse, self.coarse.prob
        tmp = self.space_transfer.restrict(eps_F)
        received = PG.u_init
        for n in range(SG.coll.num_nodes):
            CF.Reduce(self.Rcoll[n, CF.rank] * tmp, received if n == CG.rank else None, root=n, op=MPI.SUM)

        SG.eps_in = [SG._to_work(PG, received)]
        SG.delta_acc = None
        self.coarse.u0_reference = PG.dtype_u(self.coarse.u[0])
        return None

    def prolong(self):
        """
        Add the coarse level's accumulated correction to the fine level.

        Returns
        -------
        None
        """
        if self.coarse.sweep.delta_acc is None:
            return super().prolong()

        F, PF = self.fine, self.fine.prob
        SF, CF, CG = self.fine.sweep, self.comm_fine, self.comm_coarse
        SF._delta_setup()
        tmp = self.space_transfer.prolong(self.coarse.sweep.delta_acc[0])

        correction = PF.u_init
        for n in range(SF.coll.num_nodes):
            CG.Reduce(self.Pcoll[n, CG.rank] * tmp, correction if n == CF.rank else None, root=n, op=MPI.SUM)
        # quantised, then back to backend units before it is applied -- see the serial transfer
        applied = SF._to_backend(PF, SF._to_work(PF, correction))
        correction = SF._to_work(PF, applied)

        rank = CF.rank
        t_node = F.time + F.dt * SF.coll.nodes[rank]
        u_old, f_old = PF.dtype_u(F.u[rank + 1]), PF.dtype_f(F.f[rank + 1])
        F.u[rank + 1] += applied
        F.f[rank + 1] = PF.eval_f(F.u[rank + 1], t_node)

        if SF.eps_in is not None:
            # records into SF._dfs, which is what advance_residual reads
            SF._f_increment(PF, F.f[rank + 1], f_old, u_old, applied, t_node)
            SF.eps_in = SF.advance_residual(SF.eps_in, [correction], SF._dfs)
            SF.accumulate([correction])
        return None
