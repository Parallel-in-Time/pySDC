# coding=utf-8
import numpy as np
import scipy.linalg as LA
import scipy.sparse as sprs
from pySDC.Sweeper import sweeper

class generic_matrix(sweeper):
    """
    Constructs two precondition-matrices for implicit and explicit part of the equation,
    then use it for a linear iterative solver. Only for theoretical considerations useful.
    A special problem class is necessary, especially f(u) = Au

    Attributes:
        P: precondition-matrix for explicit part
        P_inv: the inverse of P
    """

    def __init__(self,coll):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        super(generic_matrix, self).__init__(coll)

        # IMEX integration matrices
        self.Qd = self.__get_Qd

        # construct preconditioner from inverse of parts of the system matrix
        # the preconditioner is interchangeble
        self.__P = sprs.kron(LA.inv(self.QE), LA.inv(self.level.prob.system_matrix))



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


    @property
    def P_inv(self):
        """
        The preconditioner
        :return:
        """
        if self.__P_inv is None:
            self.__P_inv = LA.inv(self.__P)
        return self.__P_inv


    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self,val):
        self.__P = val

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


    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        # put everything into a long vector to use simple matrix vector multiplication
        tau = np.kron(np.ones(M), L.tau.values)
        u = np.concatenate(*map(lambda x: x.values, L.u))
        u_0 = np.kron(np.ones(M), L.u[0].values)
        QkronA = sprs.kron(self.coll.Qmat, P.system_matrix)
        u_new = np.zeros(u.shape)

        # only if the level has been touched before
        assert L.status.unlocked

        # compute one iteration
        u_new[:] = u + np.dot(self.P_inv, u_0 + tau - QkronA.dot(u))

        # put the new values back into level
        L.u = np.split(u_new, M)

        # indicate presence of new values at this level
        L.status.updated = True

        return None
