from mpi4py import MPI

from pySDC.core.Errors import UnlockError
from pySDC.core.BaseTransfer import base_transfer


class base_transfer_MPI(base_transfer):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_fine = self.fine.sweep.comm
        self.comm_coarse = self.coarse.sweep.comm

        if (
            self.comm_fine.size != self.fine.sweep.coll.num_nodes
            or self.comm_coarse.size != self.coarse.sweep.coll.num_nodes
        ):
            raise NotImplementedError(
                f'{type(self).__name__} only works when each rank administers one collocation node so far!'
            )

    def restrict(self):
        """
        Space-time restriction routine

        The routine applies the spatial restriction operator to the fine values on the fine nodes, then reevaluates f
        on the coarse level. This is used for the first part of the FAS correction tau via integration. The second part
        is the integral over the fine values, restricted to the coarse level. Finally, possible tau corrections on the
        fine level are restricted as well.
        """

        F, G = self.fine, self.coarse
        CF, CG = self.comm_fine, self.comm_coarse
        SG = G.sweep
        PG = G.prob

        # only if the level is unlocked at least by prediction
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        # restrict fine values in space
        tmp_u = self.space_transfer.restrict(F.u[CF.rank + 1])

        # restrict collocation values
        G.u[0] = self.space_transfer.restrict(F.u[0])
        recvBuf = [None for _ in range(SG.coll.num_nodes)]
        recvBuf[CG.rank] = PG.u_init
        for n in range(SG.coll.num_nodes):
            CF.Reduce(self.Rcoll[n, CF.rank] * tmp_u, recvBuf[CG.rank], root=n, op=MPI.SUM)
        G.u[CG.rank + 1] = recvBuf[CG.rank]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0], G.time)
        G.f[CG.rank + 1] = PG.eval_f(G.u[CG.rank + 1], G.time + G.dt * SG.coll.nodes[CG.rank])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part in space
        tmp_tau = self.space_transfer.restrict(tauF)

        # restrict fine level tau correction part in collocation
        tauFG = tmp_tau.copy()
        for n in range(SG.coll.num_nodes):
            recvBuf = tauFG if n == CG.rank else None
            CF.Reduce(self.Rcoll[n, CF.rank] * tmp_tau, recvBuf, root=n, op=MPI.SUM)

        # build tau correction
        G.tau[CG.rank] = tauFG - tauG

        if F.tau[CF.rank] is not None:
            tmp_tau = self.space_transfer.restrict(F.tau[CF.rank])

            # restrict possible tau correction from fine in collocation
            recvBuf = [None for _ in range(SG.coll.num_nodes)]
            recvBuf[CG.rank] = PG.u_init
            for n in range(SG.coll.num_nodes):
                CF.Reduce(self.Rcoll[n, CF.rank] * tmp_tau, recvBuf[CG.rank], root=n, op=MPI.SUM)
            G.tau[CG.rank] += recvBuf[CG.rank]
        else:
            pass

        # save u and rhs evaluations for interpolation
        G.uold[CG.rank + 1] = PG.dtype_u(G.u[CG.rank + 1])
        G.fold[CG.rank + 1] = PG.dtype_f(G.f[CG.rank + 1])

        # works as a predictor
        G.status.unlocked = True

        return None

    def prolong(self):
        """
        Space-time prolongation routine

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F, G = self.fine, self.coarse
        CF, CG = self.comm_fine, self.comm_coarse
        SF = F.sweep
        PF = F.prob

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # interpolate values in space first
        tmp_u = self.space_transfer.prolong(G.u[CF.rank + 1] - G.uold[CF.rank + 1])

        # interpolate values in collocation
        recvBuf = [None for _ in range(SF.coll.num_nodes)]
        recvBuf[CF.rank] = F.u[CF.rank + 1].copy()
        for n in range(SF.coll.num_nodes):

            CG.Reduce(self.Pcoll[n, CG.rank] * tmp_u, recvBuf[n], root=n, op=MPI.SUM)
        F.u[CF.rank + 1] += recvBuf[CF.rank]

        # re-evaluate f on fine level
        F.f[CF.rank + 1] = PF.eval_f(F.u[CF.rank + 1], F.time + F.dt * SF.coll.nodes[CF.rank])

        return None

    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F, G = self.fine, self.coarse
        CF, CG = self.comm_fine, self.comm_coarse
        SF = F.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # interpolate values in space first
        tmp_u = self.space_transfer.prolong(G.u[CF.rank + 1] - G.uold[CF.rank + 1])
        tmp_f = self.space_transfer.prolong(G.f[CF.rank + 1] - G.fold[CF.rank + 1])

        # interpolate values in collocation
        recvBuf_u = [None for _ in range(SF.coll.num_nodes)]
        recvBuf_f = [None for _ in range(SF.coll.num_nodes)]
        recvBuf_u[CF.rank] = F.u[CF.rank + 1].copy()
        recvBuf_f[CF.rank] = F.f[CF.rank + 1].copy()
        for n in range(SF.coll.num_nodes):

            CG.Reduce(self.Pcoll[n, CG.rank] * tmp_u, recvBuf_u[CF.rank], root=n, op=MPI.SUM)
            CG.Reduce(self.Pcoll[n, CG.rank] * tmp_f, recvBuf_f[CF.rank], root=n, op=MPI.SUM)

        F.u[CF.rank + 1] += recvBuf_u[CF.rank]
        F.f[CF.rank + 1] += recvBuf_f[CF.rank]

        return None
