
import numpy as np

from pySDC.sweeper_classes.generic_implicit import generic_implicit



class linearized_implicit_parallel(generic_implicit):
    """
    Custom sweeper class, implements Sweeper.py

    Generic implicit sweeper, expecting lower triangular matrix QI as input

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            coll: collocation object
        """

        # call parent's initialization routine
        assert 'fixed_time_in_jacobian' in params
        super(linearized_implicit_parallel,self).__init__(params)

        # self.D, self.V = np.linalg.eig(self.coll.Qmat[1:,1:])
        self.D, self.V = np.linalg.eig(self.QI[1:, 1:])
        self.Vi = np.linalg.inv(self.V)
        # print(self.V)
        # print(self.D)

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

        dfdu = []
        for m in range(M+1):
            dfdu.append( P.eval_jacobian(L.u[m]) )

        Gu = self.integrate()
        for m in range(M):
            Gu[m] -= L.u[m + 1] - L.u[0]

        Guv = []
        for m in range(M):
            Guv.append(P.dtype_u(P.init, val=0))
            for j in range(M):
                Guv[m] += self.Vi[m, j] * Gu[j]

        # hell yeah, this is parallel!!
        uv = []
        for m in range(M):
            uv.append(
                P.solve_system_jacobian(dfdu[m], Guv[m], L.dt * self.D[m], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]))

        for m in range(M):
            for j in range(M):
                L.u[m + 1] += self.V[m, j] * uv[j]

        # hell yeah, this is parallel!!
        for m in range(M):
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None