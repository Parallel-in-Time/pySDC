import logging
import scipy.sparse as sp
import numpy as np

from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.core.Errors import UnlockError


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):
        self.finter = False
        for k, v in pars.items():
            setattr(self, k, v)

        self._freeze()


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

        self.params = _Pars(base_transfer_params)

        # set up logger
        self.logger = logging.getLogger('transfer')

        # just copy by object
        self.fine = fine_level
        self.coarse = coarse_level

        fine_grid = self.fine.sweep.coll.nodes
        coarse_grid = self.coarse.sweep.coll.nodes

        if len(fine_grid) == len(coarse_grid):
            self.Pcoll = sp.eye(len(fine_grid)).toarray()
            self.Rcoll = sp.eye(len(fine_grid)).toarray()
        else:
            self.Pcoll = self.get_transfer_matrix_Q(fine_grid, coarse_grid)
            self.Rcoll = self.get_transfer_matrix_Q(coarse_grid, fine_grid)

        # set up spatial transfer
        self.space_transfer = space_transfer_class(fine_prob=self.fine.prob, coarse_prob=self.coarse.prob,
                                                   params=space_transfer_params)

    @staticmethod
    def get_transfer_matrix_Q(f_nodes, c_nodes):
        """
        Helper routine to quickly define transfer matrices between sets of nodes (fully Lagrangian)
        Args:
            f_nodes: fine nodes
            c_nodes: coarse nodes

        Returns:
            matrix containing the interpolation weights
        """
        nnodes_f = len(f_nodes)
        nnodes_c = len(c_nodes)

        tmat = np.zeros((nnodes_f, nnodes_c))

        for i in range(nnodes_f):
            xi = f_nodes[i]
            for j in range(nnodes_c):
                den = 1.0
                num = 1.0
                for k in range(nnodes_c):
                    if k == j:
                        continue
                    else:
                        den *= c_nodes[j] - c_nodes[k]
                        num *= xi - c_nodes[k]
                tmat[i, j] = num / den

        return tmat

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
        for n in range(1, SG.coll.num_nodes + 1):
            G.u[n] = self.Rcoll[n - 1, 0] * tmp_u[0]
            for m in range(1, SF.coll.num_nodes):
                G.u[n] += self.Rcoll[n - 1, m] * tmp_u[m]

        # re-evaluate f on coarse level
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
        for n in range(1, SG.coll.num_nodes + 1):
            tauFG.append(self.Rcoll[n - 1, 0] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n - 1, m] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauFG[m] - tauG[m]

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
        for m in range(1, SG.coll.num_nodes + 1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        G.u[0] = self.space_transfer.restrict(F.u[0])
        G.f[0] = PG.eval_f(G.u[0], G.time)

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

        # interpolate values in space first
        tmp_u = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))

        # interpolate values in collocation
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(SG.coll.num_nodes):
                F.u[n] += self.Pcoll[n - 1, m] * tmp_u[m]

        # re-evaluate f on fine level
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

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # interpolate values in space first
        tmp_u = []
        tmp_f = []
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))
            tmp_f.append(self.space_transfer.prolong(G.f[m] - G.fold[m]))

        # interpolate values in collocation
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(SG.coll.num_nodes):
                F.u[n] += self.Pcoll[n - 1, m] * tmp_u[m]
                F.f[n] += self.Pcoll[n - 1, m] * tmp_f[m]

        return None
