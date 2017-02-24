import numpy as np
import logging
from scipy import interpolate

from pySDC.helpers.pysdc_helper import FrozenClass
import pySDC.helpers.transfer_helper as th
from pySDC.core.Errors import UnlockError


class base_transfer(object):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def __init__(self, fine_level, coarse_level, base_transfer_params, space_transfer_class, space_transfer_params):
        """
        Initialization routine

        Args:
            fine_level (pySDC.Level.level): fine level connected with the base_transfer operations
            coarse_level (pySDC.Level.level): coarse level connected with the base_transfer operations
            base_transfer_params (dict): parameters for the base_transfer operations
            space_transfer_class: class to perform spatial transfer
            space_transfer_params (dict): parameters for the space_transfer operations
        """

        # short helper class to add params as attributes
        class __Pars(FrozenClass):
            def __init__(self, pars):
                self.finter = False
                for k, v in pars.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = __Pars(base_transfer_params)

        # set up logger
        self.logger = logging.getLogger('transfer')

        # just copy by object
        self.fine = fine_level
        self.coarse = coarse_level

        fine_grid = self.fine.sweep.coll.nodes
        coarse_grid = self.coarse.sweep.coll.nodes

        M_fine = self.fine.sweep.coll.num_nodes
        M_coarse = self.coarse.sweep.coll.num_nodes

        weights_fine = self.fine.sweep.coll.weights
        weights_coarse = self.coarse.sweep.coll.weights

        Lagrange = [interpolate.lagrange(coarse_grid, np.eye(M_coarse)[i, :]) for i in range(M_coarse)]

        Rcoll = np.zeros((M_coarse, M_fine))
        for i in range(M_coarse):
            for j in range(M_fine):
                Rcoll[i, j] = weights_fine[j] / weights_coarse[i] * np.polyval(Lagrange[i], fine_grid[j])

        Pcoll = np.zeros((M_fine, M_coarse))
        for i in range(M_fine):
            for j in range(M_coarse):
                Pcoll[i, j] = np.polyval(Lagrange[j], fine_grid[i])

        # print(Rcoll)

        # set up preliminary transfer matrices for Q-based coarsening
        # Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=2, pad=0).toarray()
        Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=len(fine_grid), pad=0).toarray()
        #
        # print(Rcoll)
        # exit()

        # pad transfer matrices if necessary #TODO: make me go away
        # self.Pcoll = np.zeros((self.fine.sweep.coll.num_nodes + 1, self.coarse.sweep.coll.num_nodes + 1))
        # self.Rcoll = np.zeros((self.coarse.sweep.coll.num_nodes + 1, self.fine.sweep.coll.num_nodes + 1))
        # self.Pcoll[1:, 1:] = Pcoll
        # self.Rcoll[1:, 1:] = Rcoll

        self.Rcoll = Rcoll
        self.Pcoll = Pcoll

        # set up spatial transfer
        self.space_transfer = space_transfer_class(fine_prob=self.fine.prob, coarse_prob=self.coarse.prob,
                                                   params=space_transfer_params)

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
        for n in range(SG.coll.num_nodes):
            G.u[n+1] = self.Rcoll[n, 0] * tmp_u[0]
            for m in range(1,SF.coll.num_nodes):
                G.u[n+1] += self.Rcoll[n, m] * tmp_u[m]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part in space
        tmp_tau = []
        for m in range(SF.coll.num_nodes):
            tmp_tau.append(self.space_transfer.restrict(tauF[m]))

        # restrict fine level tau correction part in collocation
        tauFG = []
        for n in range(SG.coll.num_nodes):
            tauFG.append(self.Rcoll[n, 0] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n, m] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauFG[m] - tauG[m]

        if F.tau is not None:
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
        # G.uold[0] = self.space_transfer.restrict(F.u[0])

        # interpolate values in space first
        # tmp_u = [self.space_transfer.prolong(G.u[0] - G.uold[0])]
        tmp_u = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))

        # interpolate values in collocation
        # F.u[0] += tmp_u[0]
        for n in range(SF.coll.num_nodes):
            for m in range(SG.coll.num_nodes):
                F.u[n+1] += self.Pcoll[n, m] * tmp_u[m]

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
        # G.uold[0] = self.space_transfer.restrict(F.u[0])
        # G.fold[0] = PG.eval_f(G.uold[0], G.time)

        # interpolate values in space first
        # tmp_u = [self.space_transfer.prolong(G.u[0] - G.uold[0])]
        # tmp_f = [self.space_transfer.prolong(G.f[0] - G.fold[0])]
        tmp_u = []
        tmp_f = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))
            tmp_f.append(self.space_transfer.prolong(G.f[m] - G.fold[m]))

        # interpolate values in collocation
        # F.u[0] += tmp_u[0]
        # F.f[0] += tmp_f[0]
        for n in range(SF.coll.num_nodes):
            for m in range(SG.coll.num_nodes):
                F.u[n+1] += self.Pcoll[n, m] * tmp_u[m]
                F.f[n+1] += self.Pcoll[n, m] * tmp_f[m]

        return None
