import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class swfw_scalar(ptype):
    """
    Example implementing fast-wave-slow-wave scalar problem

    Attributes:
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, lambda_s=-1, lambda_f=-1000, u0=1):
        """
        Initialization routine
        """
        init = ([lambda_s.size, lambda_f.size], None, np.dtype('complex128'))
        super().__init__(init)
        self._makeAttributeAndRegister('lambda_s', 'lambda_f', 'u0', localVars=locals(), readOnly=True)

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple im=nversion of (1-dt*lambda)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                me[i, j] = rhs[i, j] / (1.0 - factor * self.lambda_f[j])

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            explicit part of RHS
        """

        fexpl = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fexpl[i, j] = self.lambda_s[i] * u[i, j]
        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fimpl[i, j] = self.lambda_f[j] * u[i, j]

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                me[i, j] = self.u0 * np.exp((self.lambda_f[j] + self.lambda_s[i]) * t)
        return me
