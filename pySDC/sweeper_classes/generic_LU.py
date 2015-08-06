import scipy.linalg as LA
import numpy as np

from pySDC.Sweeper import sweeper



class generic_LU(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    LU sweeper using LU decomposition of the Q matrix for the base integrator

    Attributes:
        Qd: U^T of Q^T = L*U
    """

    def __init__(self,params):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        super(generic_LU,self).__init__(params)

        # LU integration matrix
        self.Qd = self.__get_Qd()
        pass

    def __get_Qd(self):
        """
        Compute LU decomposition of Q^T

        Returns:
            Qd: U^T of Q^T = L*U
        """

        # strip Qmat by initial value u0
        QT = self.coll.Qmat[1:,1:].T
        # do LU decomposition of QT
        [P,L,U] = LA.lu(QT,overwrite_a=True)
        # enrich QT by initial value u0
        Qd = np.zeros(np.shape(self.coll.Qmat))
        Qd[1:,1:] = U.T
        return Qd

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1,self.coll.num_nodes+1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init,val=0))
            for j in range(1,self.coll.num_nodes+1):
                me[-1] += L.dt*self.coll.Qmat[m,j]*L.f[j]

        return me


    def update_nodes(self,level=None,stopit=None):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(M+1):
                integral[m] -= L.dt*self.Qd[m+1,j]*L.f[j]

            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0,M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m+1):
                rhs += L.dt*self.Qd[m+1,j]*L.f[j]

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m+1] = P.solve_system(rhs,L.dt*self.Qd[m+1,m+1],L.u[m+1],L.time+L.dt*self.coll.nodes[m])
            # update function values
            L.f[m+1] = P.eval_f(L.u[m+1],L.time+L.dt*self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None


    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here might be a simple copy from u[M] (if right point is a collocation node) or
        a full evaluation of the Picard formulation (if right point is not a collocation node)
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point (flag is set in collocation class)
        if self.coll.right_is_node:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt*self.coll.weights[m]*L.f[m+1]
            # add up tau correction of the full interval (last entry)
            if L.tau is not None:
                L.uend += L.tau[-1]

        return None