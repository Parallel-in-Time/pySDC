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

    def __init__(self,coll):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        super(generic_LU,self).__init__(coll)

        # LU integration matrix
        self.Qd = self.__get_Qd(coll)
        pass

    def __get_Qd(self,coll):
        """
        Compute LU decomposition of Q^T

        Args:
            coll: collocation object
        Returns:
            Qd: U^T of Q^T = L*U
        """

        # strip Qmat by initial value u0
        QT = coll.Qmat[1:,1:].T
        # do LU decomposition of QT
        [P,L,U] = LA.lu(QT,overwrite_a=True)
        # enrich QT by initial value u0
        Qd = np.zeros(np.shape(coll.Qmat))
        Qd[1:,1:] = U.T
        return Qd

    def integrate(self,weights):
        """
        Integrates the right-hand side

        Args:
            weights: integration weights, length num_nodes
        Returns:
            dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # create new instance of dtype_u, initialize values with 0
        me = P.dtype_u(P.init,val=0)

        # integrate RHS over all collocation nodes
        for j in range(self.coll.num_nodes):
            me += L.dt*weights[j]*L.f[j+1]

        return me


    def update_nodes(self):
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

        integral = []

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau
        for m in range(M):
            # get QF(u^k)_m - QdF(u^k)
            integral.append(self.integrate(self.coll.Qmat[m+1,1:] - self.Qd[m+1,1:]))
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
            L.u[m+1] = P.solve_system(rhs,L.dt*self.Qd[m+1,m+1],L.u[m+1])
            # update function values
            L.f[m+1] = P.eval_f(L.u[m+1],L.time+L.dt*self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None