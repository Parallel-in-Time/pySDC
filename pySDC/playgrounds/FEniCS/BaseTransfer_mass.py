import logging
import scipy.sparse as sp
import numpy as np

from pySDC.core.Errors import UnlockError

from pySDC.core.BaseTransfer import base_transfer


class base_transfer_mass(base_transfer):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def restrict(self):
        """
        Space-time restriction routine

        The routine applies the spatial restriction operator to teh fine values on the fine nodes, then reevaluates f
        on the coarse level. This is used for the first part of the FAS correction tau via integration. The second part
        is the integral over the fine values, restricted to the coarse level. Finally, possible tau corrections on the
        fine level are restricted as well.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PG = G.prob
        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only if the level is unlocked at least by prediction
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        # restrict fine values in space
        tmp_u = []
        for m in range(1, SF.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.restrict(F.u[m]))

        # restrict collocation values
        G.u[0] = self.space_transfer.restrict(F.u[0])
        for n in range(1, SG.coll.num_nodes + 1):
            G.u[n] = self.Rcoll[n - 1, 0] * tmp_u[0]
            for m in range(1, SF.coll.num_nodes):
                G.u[n] += self.Rcoll[n - 1, m] * tmp_u[m]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        for m in range(SG.coll.num_nodes):
            tauG[m] = PG.apply_mass_matrix(G.u[m + 1]) - tauG[m]

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        for m in range(SF.coll.num_nodes):
            tauF[m] = PF.apply_mass_matrix(F.u[m + 1]) - tauF[m]

        # restrict fine level tau correction part in space
        tmp_tau = []
        for m in range(SF.coll.num_nodes):
            tmp_tau.append(self.space_transfer.restrict(tauF[m]))

        # restrict fine level tau correction part in collocation
        tauFG = []
        for n in range(1, SG.coll.num_nodes + 1):
            tauFG.append(self.Rcoll[n - 1, 0] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n - 1, m] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauG[m] - tauFG[m]

        if F.tau[0] is not None:
            # restrict possible tau correction from fine in space
            tmp_tau = []
            for m in range(SF.coll.num_nodes):
                tmp_tau.append(self.space_transfer.restrict(F.tau[m]))

            # restrict possible tau correction from fine in collocation
            for n in range(SG.coll.num_nodes):
                for m in range(SF.coll.num_nodes):
                    G.tau[n] += self.Rcoll[n, m] * tmp_tau[m]
        else:
            pass

        # save u and rhs evaluations for interpolation
        for m in range(SG.coll.num_nodes + 1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        G.u[0] = self.space_transfer.restrict(PF.apply_mass_matrix(F.u[0]))

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
        F = self.fine
        G = self.coarse

        PF = F.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # we need to update u0 here for the predictor step, since here the new values for the fine sweep are not
        # received from the previous processor but interpolated from the coarse level.
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        G.uold[0] = self.space_transfer.restrict(F.u[0])

        # interpolate values in space first
        tmp_u = [self.space_transfer.prolong(G.u[0] - G.uold[0])]
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))

        # interpolate values in collocation
        # F.u[0] += tmp_u[0]
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(1, SG.coll.num_nodes + 1):
                F.u[n] += self.Pcoll[n - 1, m - 1] * tmp_u[m]

        # re-evaluate f on fine level
        # F.f[0] = PF.eval_f(F.u[0], F.time)
        for m in range(1, SF.coll.num_nodes + 1):
            F.f[m] = PF.eval_f(F.u[m], F.time + F.dt * SF.coll.nodes[m - 1])

        return None

    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        G.uold[0] = self.space_transfer.restrict(F.u[0])
        G.fold[0] = PG.eval_f(G.uold[0], G.time)

        # interpolate values in space first
        tmp_u = [self.space_transfer.prolong(G.u[0] - G.uold[0])]
        tmp_f = [self.space_transfer.prolong(G.f[0] - G.fold[0])]
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))
            tmp_f.append(self.space_transfer.prolong(G.f[m] - G.fold[m]))

        # interpolate values in collocation
        F.u[0] += tmp_u[0]
        F.f[0] += tmp_f[0]
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(1, SG.coll.num_nodes + 1):
                F.u[n] += self.Pcoll[n - 1, m - 1] * tmp_u[m]
                F.f[n] += self.Pcoll[n - 1, m - 1] * tmp_f[m]

        return None
