import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher


# noinspection PyUnusedLocal
class generalized_fisher_jac(generalized_fisher):

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([self.params.interval[0] + (i + 1) * self.dx for i in range(self.params.nvars)])

        for i in range(self.params.nvars):
                r2 = xvalues[i] ** 2
                me.values[i] = 1.0/2.0 + 1.0/2.0 * np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))
        return me

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        # noinspection PyTypeChecker
        dfdu = self.A[1:-1, 1:-1] + sp.diags(self.params.lambda0 ** 2 - self.params.lambda0 ** 2 *
                                             (self.params.nu + 1) * u.values ** self.params.nu, offsets=0)
        # print(type(dfdu))
        return dfdu
