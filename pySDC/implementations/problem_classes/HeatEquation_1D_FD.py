import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh


# noinspection PyUnusedLocal
class heat1d(ptype):
    """
    Example implementing the unforced 1D heat equation with Dirichlet-0 BC in [0,1],
    discretized using central finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'freq']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars'] + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d, self).__init__(init=(problem_params['nvars'], None, np.dtype('float64')),
                                     dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1.0 / (self.params.nvars + 1)
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx)

    @staticmethod
    def __get_A(N, nu, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil, [-1, 0, 1], shape=(N, N), format='csc')
        A *= nu / (dx ** 2)
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
        me[:] = spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.A, rhs)
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
        if self.params.freq >= 0:
            xvalues = np.array([(i + 1) * self.dx for i in range(self.params.nvars)])
            rho = (2.0 - 2.0 * np.cos(np.pi * self.params.freq * self.dx)) / self.dx ** 2
            me[:] = np.sin(np.pi * self.params.freq * xvalues) * \
                np.exp(-t * self.params.nu * rho)
        else:
            np.random.seed(1)
            me[:] = np.random.rand(self.params.nvars)
        return me
