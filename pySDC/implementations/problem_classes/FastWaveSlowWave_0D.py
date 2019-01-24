
import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.complex_mesh import mesh, rhs_imex_mesh


# noinspection PyUnusedLocal
class swfw_scalar(ptype):
    """
    Example implementing fast-wave-slow-wave scalar problem

    Attributes:
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=rhs_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['lambda_s', 'lambda_f', 'u0']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        init = (problem_params['lambda_s'].size, problem_params['lambda_f'].size)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(swfw_scalar, self).__init__(init, dtype_u, dtype_f, problem_params)

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
        for i in range(self.params.lambda_s.size):
            for j in range(self.params.lambda_f.size):
                me.values[i, j] = rhs.values[i, j] / (1.0 - factor * self.params.lambda_f[j])

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
        for i in range(self.params.lambda_s.size):
            for j in range(self.params.lambda_f.size):
                fexpl.values[i, j] = self.params.lambda_s[i] * u.values[i, j]
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
        for i in range(self.params.lambda_s.size):
            for j in range(self.params.lambda_f.size):
                fimpl.values[i, j] = self.params.lambda_f[j] * u.values[i, j]

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
        for i in range(self.params.lambda_s.size):
            for j in range(self.params.lambda_f.size):
                me.values[i, j] = self.params.u0 * np.exp((self.params.lambda_f[j] + self.params.lambda_s[i]) * t)
        return me
