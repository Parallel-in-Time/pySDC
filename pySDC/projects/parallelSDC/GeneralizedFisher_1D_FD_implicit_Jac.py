import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher


# noinspection PyUnusedLocal
class generalized_fisher_jac(generalized_fisher):
    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        # noinspection PyTypeChecker
        dfdu = self.A[1:-1, 1:-1] + sp.diags(
            self.params.lambda0**2 - self.params.lambda0**2 * (self.params.nu + 1) * u**self.params.nu, offsets=0
        )

        return dfdu

    def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            dfdu: the Jacobian of the RHS of the ODE
            rhs: right-hand side for the linear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = self.dtype_u((self.init[0], self.init[1], np.dtype('complex128')))
        me[:] = spsolve(sp.eye(self.params.nvars) - factor * dfdu, rhs)
        return me
