import abc
import numpy as np
import scipy.sparse as sp

from pySDC.Plugins.pysdc_helper import FrozenClass
import pySDC.Plugins.transfer_helper as th


class base_transfer(object):
    """
    Standard base_transfer class

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_level,coarse_level,base_transfer_params,space_transfer_class,space_transfer_params):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the base_transfer operations
            coarse_level: coarse level connected with the base_transfer operations
            base_transfer_params: parameters for the base_transfer operations
            space_transfer_class: class to perform spatial transfer
            space_transfer_params: parameters for the space_transfer operations
        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self,params):

                self.finter = False
                self.coll_iorder = None
                self.coll_rorder = 1

                for k,v in params.items():
                    setattr(self,k,v)

                self._freeze()

        self.params = pars(base_transfer_params)

        # just copy by object
        self.fine = fine_level
        self.coarse = coarse_level

        if not self.fine.sweep.coll.left_is_node:
            fine_grid = np.concatenate(([0], self.fine.sweep.coll.nodes))
            coarse_grid = np.concatenate(([0], self.coarse.sweep.coll.nodes))
        else:
            fine_grid = self.fine.sweep.coll.nodes
            coarse_grid = self.coarse.sweep.coll.nodes

        if self.params.coll_iorder is None or self.params.coll_iorder > len(coarse_grid):
            print('WARNING: requested order of Q-interpolation is not valid, resetting to %s' %len(coarse_grid))
            self.params.coll_iorder = len(coarse_grid)
        if self.params.coll_rorder != 1:
            print('WARNING: requested order of Q-restriction is != 1, can lead to weird behavior!')

        Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.coll_iorder).toarray()
        Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=self.params.coll_rorder).toarray()

        if self.fine.sweep.coll.left_is_node:
            self.Pcoll = np.zeros((self.fine.sweep.coll.num_nodes + 1, self.coarse.sweep.coll.num_nodes + 1))
            self.Rcoll = self.Pcoll.T
            self.Pcoll[1:,1:] = Pcoll
            self.Rcoll[1:,1:] = Rcoll
        else:
            self.Pcoll = Pcoll
            self.Rcoll = Rcoll

        self.space_transfer = space_transfer_class(fine_prob=self.fine.prob, coarse_prob=self.coarse.prob, params=space_transfer_params)

    # FIXME: add time restriction
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

        PF = F.prob
        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction
        assert F.status.unlocked

        # restrict fine values in space
        tmp_u = [self.space_transfer.restrict(F.u[0])]
        for m in range(1,SF.coll.num_nodes+1):
            tmp_u.append(self.space_transfer.restrict(F.u[m]))

        # restrict collocation values
        G.u[0] = tmp_u[0]
        for n in range(1, SG.coll.num_nodes + 1):
            G.u[n] = self.Rcoll[n,0]*tmp_u[0]
            for m in range(1, SF.coll.num_nodes + 1):
                G.u[n] += self.Rcoll[n,m]*tmp_u[m]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0],G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m],G.time+G.dt*SG.coll.nodes[m-1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part in space
        tmp_tau = []
        for m in range(0, SF.coll.num_nodes):
            tmp_tau.append(self.space_transfer.restrict(tauF[m]))

        # restrict fine level tau correction part in collocation
        tauFG = [tmp_tau[0]]
        for n in range(1, SG.coll.num_nodes):
            tauFG.append(self.Rcoll[n+1, 1] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n+1, m+1] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauFG[m] - tauG[m]


        if F.tau is not None:
            # restrict possible tau correction from fine in space
            tmp_tau = []
            for m in range(0, SF.coll.num_nodes):
                tmp_tau.append(self.space_transfer.restrict(F.tau[m]))

            # restrict possible tau correction from fine in collocation
            for n in range(0, SG.coll.num_nodes):
                G.tau[n] += self.Rcoll[n + 1, 1] * tmp_tau[0]
                for m in range(1, SF.coll.num_nodes):
                    G.tau[n] += self.Rcoll[n + 1, m + 1] * tmp_tau[m]

        # save u and rhs evaluations for interpolation
        for m in range(SG.coll.num_nodes+1):
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
        assert G.status.unlocked

        # build coarse correction

        # we need to update u0 here for the predictor step, since here the new values for the fine sweep are not received
        # from the previous processor but interpolated from the coarse level.
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        G.uold[0] = self.space_transfer.restrict(F.u[0])

        # interpolate values in space first
        tmp_u = [self.space_transfer.prolong(G.u[0] - G.uold[0])]
        for m in range(1, SG.coll.num_nodes+1):
            tmp_u.append(self.space_transfer.prolong(G.u[m] - G.uold[m]))

        # interpolate values in collocation
        F.u[0] += tmp_u[0]
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(0, SG.coll.num_nodes+1):
                F.u[n] += self.Pcoll[n, m] * tmp_u[m]

        # re-evaluate f on fine level
        F.f[0] = PF.eval_f(F.u[0], F.time)
        for m in range(1,SF.coll.num_nodes+1):
            F.f[m] = PF.eval_f(F.u[m],F.time+F.dt*SF.coll.nodes[m-1])

        return None


    # FIXME: add time prolongation
    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob
        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only of the level is unlocked at least by prediction or restriction
        assert G.status.unlocked
        # can only do space-restriction so far
        assert np.array_equal(SF.coll.nodes,SG.coll.nodes)

        # build coarse correction
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        G.uold[0] = self.space_transfer.restrict(F.u[0])
        G.fold[0] = PG.eval_f(G.uold[0],G.time)

        for m in range(0,SF.coll.num_nodes+1):
            F.u[m] += self.space_transfer.prolong(G.u[m] - G.uold[m])
            F.f[m] += self.space_transfer.prolong(G.f[m] - G.fold[m])

        return None
