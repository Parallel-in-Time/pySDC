import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class linearized_implicit_parallel(generic_implicit):
    """
    Parallel sweeper using Newton for linearization
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'fixed_time_in_jacobian' not in params:
            params['fixed_time_in_jacobian'] = 0

        # call parent's initialization routine
        super(linearized_implicit_parallel, self).__init__(params)

        self.D, self.V = np.linalg.eig(self.QI[1:, 1:])
        self.Vi = np.linalg.inv(self.V)

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

        # form Jacobian on each node
        dfdu = []
        for m in range(M + 1):
            dfdu.append(P.eval_jacobian(L.u[m]))

        # form collocation problem
        Gu = self.integrate()
        for m in range(M):
            Gu[m] -= L.u[m + 1] - L.u[0]

        # transform collocation problem forward
        Guv = []
        for m in range(M):
            Guv.append(P.dtype_u((P.init[0], P.init[1], np.dtype('complex128')), val=0.0+0.0j))
            for j in range(M):
                Guv[m] += self.Vi[m, j] * Gu[j]

        # solve implicit system with Jacobians
        uv = []
        for m in range(M):  # hell yeah, this is parallel!!
            uv.append(P.solve_system_jacobian(dfdu[m], Guv[m], L.dt * self.D[m], L.u[m + 1],
                                              L.time + L.dt * self.coll.nodes[m]))

        # transform solution backward
        for m in range(M):
            tmp = P.dtype_u((P.init[0], P.init[1], np.dtype('complex128')), val=0.0 + 0.0j)
            for j in range(M):
                tmp += self.V[m, j] * uv[j]
            L.u[m + 1][:] += np.real(tmp)

        # evaluate f
        for m in range(M):  # hell yeah, this is parallel!!
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
