import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class testequation0d(ptype):
    """
    Example implementing a bundle of test equations at once (via diagonal matrix)

    Attributes:
        A: digonal matrix containing the parameters
    """

    dtype_u = mesh
    dtype_f = mesh

    # TODO : add default values
    def __init__(self, lambdas=1, u0=0.0):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """
        assert not any(isinstance(i, list) for i in lambdas), 'ERROR: expect flat list here, got %s' % lambdas
        nvars = len(lambdas)
        assert nvars > 0, 'ERROR: expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('complex128')))

        self.A = self.__get_A(lambdas)
        self._makeAttributeAndRegister('nvars', 'lambdas', 'u0', localVars=locals(), readOnly=True)
        self.work_counters['rhs'] = WorkCounter()

    @staticmethod
    def __get_A(lambdas):
        """
        Helper function to assemble FD matrix A in sparse format.

        Parameters
        ----------
        lambdas : list
            List of lambda parameters.

        Returns
        -------
        scipy.sparse.csc_matrix
            Diagonal matrix A in CSC format.
        """

        A = sp.diags(lambdas)
        return A

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        L = splu(sp.eye(self.nvars, format='csc') - factor * self.A)
        me[:] = L.solve(rhs)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.testequation0d.dtype_u
            Initial solution.
        t_init : float
            The initial time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        u_init = (self.u0 if u_init is None else u_init) * 1.0
        t_init = 0.0 if t_init is None else t_init * 1.0

        me = self.dtype_u(self.init)
        me[:] = u_init * np.exp((t - t_init) * np.array(self.lambdas))
        return me
