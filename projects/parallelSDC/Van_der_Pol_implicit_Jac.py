import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol


# noinspection PyUnusedLocal
class vanderpol_jac(vanderpol):

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        x1 = u.values[0]
        x2 = u.values[1]

        dfdu = np.array([[0, 1], [-2 * self.params.mu * x1 * x2 - 1, self.params.mu * (1 - x1 ** 2)]])

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

        me = self.dtype_u(2)
        me.values = spsolve(sp.eye(2) - factor * dfdu, rhs.values)
        return me
