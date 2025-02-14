"""
These sweepers are made for use with ParaDiag. They can be used to some degree with SDC as well, but unless you know what you are doing, you probably want another sweeper.
"""

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
import numpy as np
import scipy.sparse as sp


class QDiagonalization(generic_implicit):
    """
    Sweeper solving the collocation problem directly via diagonalization of Q. Mainly made for ParaDiag.
    Can be reconfigured for use with SDC.

    Note that the initial conditions for the collocation problem are generally stored in node zero in pySDC. However,
    this sweeper is intended for ParaDiag, where a node-local residual is needed as a right hand side for this sweeper
    rather than a step local one. Therefore, this sweeper has an option `ignore_ic`. If true, the value in node zero
    will only be used in computing the step-local residual, but not in the solves. If false, the values on the nodes
    will be ignored in the solves and the node-zero value will be used as initial conditions. When using this as a time-
    parallel algorithm outside ParaDiag, you should set this parameter to false, which is not the default!

    Similarly, in ParaDiag, the solution is in Fourier space right after the solve. It therefore makes little sense to
    evaluate the right hand side directly after. By default, this is not done! Set `update_f_evals=True` in the
    parameters if you want to use this sweeper in SDC.
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        if 'G_inv' not in params.keys():
            params['G_inv'] = np.eye(params['num_nodes'])
        params['update_f_evals'] = params.get('update_f_evals', False)
        params['ignore_ic'] = params.get('ignore_ic', True)

        super().__init__(params)

        self.set_G_inv(self.params.G_inv)

    def set_G_inv(self, G_inv):
        """
        In ParaDiag, QG^{-1} is diagonalized. This function stores the G_inv matrix and computes and stores the diagonalization.
        """
        self.params.G_inv = G_inv
        self.w, self.S, self.S_inv = self.computeDiagonalization(A=self.coll.Qmat[1:, 1:] @ self.params.G_inv)

    @staticmethod
    def computeDiagonalization(A):
        """
        Compute diagonalization of dense matrix A = S diag(w) S^-1

        Args:
            A (numpy.ndarray): dense matrix to diagonalize

        Returns:
            numpy.array: Diagonal entries of the diagonalized matrix w
            numpy.ndarray: Matrix of eigenvectors S
            numpy.ndarray: Inverse of S
        """
        w, S = np.linalg.eig(A)
        S_inv = np.linalg.inv(S)
        assert np.allclose(S @ np.diag(w) @ S_inv, A)
        return w, S, S_inv

    def mat_vec(self, mat, vec):
        """
        Compute matrix-vector multiplication. Vector can be list.

        Args:
            mat: Matrix
            vec: Vector

        Returns:
            list: mat @ vec
        """
        assert mat.shape[1] == len(vec)
        result = []
        for m in range(mat.shape[0]):
            result.append(self.level.prob.u_init)
            for j in range(mat.shape[1]):
                result[-1] += mat[m, j] * vec[j]
        return result

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        if L.tau[0] is not None:
            raise NotImplementedError('This sweeper does not work with multi-level SDC')

        # perform local solves on the collocation nodes, can be parallelized!
        if self.params.ignore_ic:
            x1 = self.mat_vec(self.S_inv, [self.level.u[m + 1] for m in range(M)])
        else:
            x1 = self.mat_vec(self.S_inv, [self.level.u[0] for _ in range(M)])
        x2 = []
        for m in range(M):
            # TODO: need to put averaged x1 in u0 here for nonlinear problems
            u0 = L.u_avg[m] if L.u_avg[m] is not None else x1[m]
            x2.append(P.solve_system(x1[m], self.w[m] * L.dt, u0=u0, t=L.time + L.dt * self.coll.nodes[m]))
        z = self.mat_vec(self.S, x2)
        y = self.mat_vec(self.params.G_inv, z)

        # update solution and evaluate right hand side
        for m in range(M):
            L.u[m + 1] = y[m]
            if self.params.update_f_evals:
                raise
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        L.status.updated = True
        return None

    def eval_f_at_all_nodes(self):
        L = self.level
        P = self.level.prob
        for m in range(self.coll.num_nodes):
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

    def get_residual(self):
        """
        This function computes and returns the "spatially extended" residual, not the norm of the residual!

        Returns:
            pySDC.datatype: Spatially extended residual

        """
        self.eval_f_at_all_nodes()

        # start with integral dt*Q*F
        residual = self.integrate()

        # subtract u and add u0 to arrive at r = dt*Q*F - u + u0
        for m in range(self.coll.num_nodes):
            residual[m] -= self.level.u[m + 1]
            residual[m] += self.level.u[0]

        return residual


class QDiagonalizationIMEX(QDiagonalization):
    """
    Use as sweeper class for ParaDiag with IMEX splitting. Note that it will not work with SDC.
    """

    integrate = imex_1st_order.integrate
