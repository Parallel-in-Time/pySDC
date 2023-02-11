import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class testequation0d(ptype):
    """
    Example implementing a bundle of test equations at once (via diagonal matrix)

    Attributes:
        A: digonal matrix containing the parameters
    """
    
    # TODO : add default values
    def __init__(self, lambdas, u0):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """
        assert not any(isinstance(i, list) for i in lambdas), (
            'ERROR: expect flat list here, got %s' % lambdas
        )
        nvars = len(lambdas)
        assert nvars > 0, 'ERROR: expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(
            init=(nvars, None, np.dtype('complex128')),
            dtype_u=mesh,
            dtype_f=mesh
        )

        self.A = self.__get_A(lambdas)
        self._makeAttributeAndRegister('nvars', 'lambdas', 'u0', localVars=locals(), readOnly=True)

    @staticmethod
    def __get_A(lambdas):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            lambdas (list): list of lambda parameters

        Returns:
            scipy.sparse.csc_matrix: diagonal matrix A in CSC format
        """

        A = sp.diags(lambdas)
        return A

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        L = splu(sp.eye(self.nvars, format='csc') - factor * self.A)
        me[:] = L.solve(rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        me[:] = self.u0 * np.exp(t * np.array(self.lambdas))
        return me
