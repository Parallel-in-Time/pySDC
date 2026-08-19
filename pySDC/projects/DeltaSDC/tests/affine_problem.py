r"""
A minimal affine test problem: :math:`f(u) = A u + b` with constant :math:`b`.

Used to check that the delta-form sweeper removes the affine part before reusing the stock
``solve_system``, since :math:`f(w+\delta) - f(w) = A\delta` regardless of :math:`b`.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced


class heat_affine(heatNd_unforced):
    """Heat equation with a constant source term and a matching implicit solve."""

    B_CONST = 0.7

    def eval_f(self, u, t):
        """
        Evaluate the affine right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        dtype_f
            The right-hand side.
        """
        f = super().eval_f(u, t)
        f[:] = np.asarray(f) + self.B_CONST
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`u - factor (A u + b) = rhs`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess, unused for this direct solve.
        t : float
            Current time.

        Returns
        -------
        dtype_u
            The solution.
        """
        identity = sp.eye(self.A.shape[0], format='csc')
        b = np.asarray(rhs, dtype=np.float64).reshape(-1) + factor * self.B_CONST
        me = self.dtype_u(self.init)
        me[:] = spsolve((identity - factor * self.A).tocsc(), b).reshape(self.nvars)
        return me
