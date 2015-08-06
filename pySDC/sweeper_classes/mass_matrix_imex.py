import numpy as np
import scipy.linalg as LA
from pySDC.sweeper_classes import imex_1st_order

class mass_matrix_imex(imex_1st_order.imex_1st_order):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper, now with mass matrix and LU decomposition for the implicit term

    Attributes:
        QI: St. Martin's trick matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self,params):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        super(mass_matrix_imex,self).__init__(params)

        # IMEX integration matrices
        [self.QI, self.QE] = self.__get_Qd


    @property
    def __get_Qd(self):
        """
        Sets the integration matrices QI and QE for the IMEX sweeper

        Returns:
            QI: St. Martin's trick, use LU decomposition
            QE: explicit Euler matrix, will also act on u0
        """
        # QI = np.zeros(np.shape(self.coll.Qmat))
        QE = np.zeros(np.shape(self.coll.Qmat))
        for m in range(self.coll.num_nodes + 1):
            # QI[m, 1:m+1] = self.coll.delta_m[0:m]
            QE[m, 0:m] = self.coll.delta_m[0:m]

        # This is for using LU decomposition
        # strip Qmat by initial value u0
        QT = self.coll.Qmat[1:,1:].T
        # do LU decomposition of QT
        [P,L,U] = LA.lu(QT,overwrite_a=True)
        # enrich QT by initial value u0
        Qd = np.zeros(np.shape(self.coll.Qmat))
        Qd[1:,1:] = U.T
        QI = Qd

        return QI, QE


    def update_nodes(self,level,stopit):
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
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

         # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(M+1):
                integral[m] -= L.dt*(self.QI[m+1,j]*L.f[j].impl + self.QE[m+1,j]*L.f[j].expl)
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
                rhs += L.dt*(self.QI[m+1,j]*L.f[j].impl + self.QE[m+1,j]*L.f[j].expl)
            # apply mass matrix to rhs
            rhs = P.apply_mass_matrix(rhs)

            # implicit solve with prefactor stemming from QI
            L.u[m+1] = P.solve_system(rhs,L.dt*self.QI[m+1,m+1],L.u[m+1],L.time+L.dt*self.coll.nodes[m])
            # update function values
            L.f[m+1] = P.eval_f(L.u[m+1],L.time+L.dt*self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None