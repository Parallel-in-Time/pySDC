import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class swfw_scalar(ptype):
    r"""
    This class implements the fast-wave-slow-wave scalar problem fully investigated in _[1]. It is defined by

    .. math::
        \frac{d u(t)}{dt} = \lambda_f u(t) + \lambda_s u(t),

    where :math:`\lambda_f` denotes the part of the fast wave, and :math:`\lambda_s` is the part of the slow wave with
    :math:`\lambda_f \gg \lambda_s`. Let :math:`u_0` be the initial condition to the problem, then the exact solution
    is given by

    .. math::
        u(t) = u_0 \exp((\lambda_f + \lambda_s) t).

    Parameters
    ----------
    lambda_s : np.ndarray, optional
        Part of the slow wave.
    lambda_f : np.ndarray, optional
        Part of the fast wave.
    u0 : np.ndarray, optional
        Initial condition of the problem.

    References
    ----------
    .. [1] D. Ruprecht, R. Speck. Spectral deferred corrections with fast-wave slow-wave splitting. SIAM J. Sci. Comput. Vol. 38 No. 4 (2016).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, lambda_s=-1, lambda_f=-1000, u0=1):
        """Initialization routine"""

        init = ([lambda_s.size, lambda_f.size], None, np.dtype('complex128'))
        super().__init__(init)
        self._makeAttributeAndRegister('lambda_s', 'lambda_f', 'u0', localVars=locals(), readOnly=True)

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple im=nversion of (1-dt*lambda)u = rhs.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                me[i, j] = rhs[i, j] / (1.0 - factor * self.lambda_f[j])

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        fexpl : dtype_u
            Explicit part of right-hand side.
        """

        fexpl = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fexpl[i, j] = self.lambda_s[i] * u[i, j]
        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        fimpl : dtype_u
            Implicit part of right-hand side.
        """

        fimpl = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fimpl[i, j] = self.lambda_f[j] * u[i, j]

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                me[i, j] = self.u0 * np.exp((self.lambda_f[j] + self.lambda_s[i]) * t)
        return me
