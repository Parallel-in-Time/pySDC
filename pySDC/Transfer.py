import abc
import numpy as np
import copy as cp

from future.utils import with_metaclass


class transfer(with_metaclass(abc.ABCMeta)):
    """
    Abstract transfer class

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_level,coarse_level):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations
            coarse_level: coarse level connected with the transfer operations
        """

        # just copy by object
        self.fine = fine_level
        self.coarse = coarse_level
        self.init_c = self.coarse.prob.init
        self.init_f = self.fine.prob.init

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
        # can only do space-restriction so far
        assert np.array_equal(SF.coll.nodes,SG.coll.nodes)

        # restrict fine values in space, reevaluate f on coarse level
        G.u[0] = self.restrict_space(F.u[0])
        G.f[0] = PG.eval_f(G.u[0],G.time)
        for m in range(1,SG.coll.num_nodes+1):
            G.u[m] = self.restrict_space(F.u[m])
            G.f[m] = PG.eval_f(G.u[m],G.time+G.dt*SG.coll.nodes[m-1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part
        tauFG = []
        for m in range(SG.coll.num_nodes):
            tauFG.append(self.restrict_space(tauF[m]))

        # build tau correction, also restrict possible tau correction from fine
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauFG[m] - tauG[m]
            if F.tau is not None:
                G.tau[m] += self.restrict_space(F.tau[m])

        # save u and rhs evaluations for interpolation
        for m in range(SG.coll.num_nodes+1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        # works as a predictor
        G.status.unlocked = True

        return None

    # FIXME: add time prolongation
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
        # can only do space-restriction so far
        assert np.array_equal(SF.coll.nodes,SG.coll.nodes)

        # build coarse correction
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        F.u[0] += self.prolong_space(G.u[0] - self.restrict_space(F.u[0]))
        F.f[0] = PF.eval_f(F.u[0],F.time)

        for m in range(1,SF.coll.num_nodes+1):
            F.u[m] += self.prolong_space(G.u[m] - G.uold[m])
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
        F.u[0] += self.prolong_space(G.u[0] - self.restrict_space(F.u[0]))
        F.f[0] = PF.eval_f(F.u[0],F.time)

        for m in range(1,SF.coll.num_nodes+1):
            F.u[m] += self.prolong_space(G.u[m] - G.uold[m])
            F.f[m].impl += self.prolong_space(G.f[m].impl - G.fold[m].impl)
            F.f[m].expl += self.prolong_space(G.f[m].expl - G.fold[m].expl)

        return None


    @abc.abstractmethod
    def restrict_space(self,F):
        """
        Abstract interface for restriction in space

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        pass

    @abc.abstractmethod
    def prolong_space(self,G):
        """
        Abstract interface for prolongation in space

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        pass



