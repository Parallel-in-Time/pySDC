import numpy as np
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.implementations.problem_classes.acoustic_helpers.buildWave1DMatrix import (
    getWave1DMatrix,
    getWave1DAdvectionMatrix,
)


# noinspection PyUnusedLocal
class acoustic_1d_imex(ptype):
    """
    Example implementing the one-dimensional IMEX acoustic-advection

    TODO : doku

    Attributes
    ----------
    mesh : np.ndarray
        1d mesh.
    dx : float
        Mesh size.
    Dx : scipy.csc_matrix
        Matrix for the advection operator.
    Id : scipy.csc_matrix
        Sparse identity matrix.
    A : scipy.csc_matrix
        Matrix for the wave operator.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars, cs, cadv, order_adv, waveno):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'cs', 'cadv', 'order_adv', 'waveno', localVars=locals(), readOnly=True)

        self.mesh = np.linspace(0.0, 1.0, self.nvars[1], endpoint=False)
        self.dx = self.mesh[1] - self.mesh[0]

        self.Dx = -self.cadv * getWave1DAdvectionMatrix(self.nvars[1], self.dx, self.order_adv)
        self.Id, A = getWave1DMatrix(self.nvars[1], self.dx, ['periodic', 'periodic'], ['periodic', 'periodic'])
        self.A = -self.cs * A

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
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

        M = self.Id - factor * self.A

        b = np.concatenate((rhs[0, :], rhs[1, :]))

        sol = spsolve(M, b)

        me = self.dtype_u(self.init)
        me[0, :], me[1, :] = np.split(sol, 2)

        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        fexpl : dtype_f
            Explicit part of the right-hand side.
        """

        b = np.concatenate((u[0, :], u[1, :]))
        sol = self.Dx.dot(b)

        fexpl = self.dtype_u(self.init)
        fexpl[0, :], fexpl[1, :] = np.split(sol, 2)

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        fimpl : dtype_f
            Implicit part of the right-hand side.
        """

        b = np.concatenate((u[:][0, :], u[:][1, :]))
        sol = self.A.dot(b)

        fimpl = self.dtype_u(self.init, val=0.0)
        fimpl[0, :], fimpl[1, :] = np.split(sol, 2)

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

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

        def u_initial(x, k):
            return np.sin(k * 2.0 * np.pi * x) + np.sin(2.0 * np.pi * x)

        me = self.dtype_u(self.init)
        me[0, :] = 0.5 * u_initial(self.mesh - (self.cadv + self.cs) * t, self.waveno) - 0.5 * u_initial(
            self.mesh - (self.cadv - self.cs) * t, self.waveno
        )
        me[1, :] = 0.5 * u_initial(self.mesh - (self.cadv + self.cs) * t, self.waveno) + 0.5 * u_initial(
            self.mesh - (self.cadv - self.cs) * t, self.waveno
        )
        return me
