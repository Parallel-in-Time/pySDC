import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class advection1d(ptype):
    """
    Example implementing the unforced 1D advection equation with periodic BC in [0,1],
    discretized using upwinding finite differences

    Attributes:
        A: FD discretization of the gradient operator using upwinding
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'c', 'freq']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')
        if problem_params['freq'] >= 0 and problem_params['freq'] % 2 != 0:
            raise ProblemError('need even number of frequencies due to periodic BCs')

        if 'order' not in problem_params:
            problem_params['order'] = 1
        if 'type' not in problem_params:
            problem_params['type'] = 'upwind'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advection1d, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.params.nvars
        self.A = self.__get_A(self.params.nvars, self.params.c, self.dx, self.params.order, self.params.type)

    @staticmethod
    def __get_A(N, c, dx, order, type):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            c (float): diffusion coefficient
            dx (float): distance between two spatial nodes
            order (int): specifies order of discretization
            type (string): upwind or centered differences

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        coeff = None
        stencil = None
        zero_pos = None

        if type == 'center':

            if order == 2:
                stencil = [-1.0, 0.0, 1.0]
                zero_pos = 2
                coeff = 1.0 / 2.0
            elif order == 4:
                stencil = [1.0, -8.0, 0.0, 8.0, -1.0]
                zero_pos = 3
                coeff = 1.0 / 12.0
            elif order == 6:
                stencil = [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]
                zero_pos = 4
                coeff = 1.0 / 60.0
            else:
                raise ProblemError("Order " + str(order) + " not implemented.")

        else:

            if order == 1:
                stencil = [-1.0, 1.0]
                coeff = 1.0
                zero_pos = 2

            elif order == 2:
                stencil = [1.0, -4.0, 3.0]
                coeff = 1.0 / 2.0
                zero_pos = 3

            elif order == 3:
                stencil = [1.0, -6.0, 3.0, 2.0]
                coeff = 1.0 / 6.0
                zero_pos = 3

            elif order == 4:
                stencil = [-5.0, 30.0, -90.0, 50.0, 15.0]
                coeff = 1.0 / 60.0
                zero_pos = 4

            elif order == 5:
                stencil = [3.0, -20.0, 60.0, -120.0, 65.0, 12.0]
                coeff = 1.0 / 60.0
                zero_pos = 5
            else:
                raise ProblemError("Order " + str(order) + " not implemented.")

        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(
            (
                [N - i - 1 for i in reversed(range(zero_pos - 1))],
                [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))],
            )
        )
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N))

        A = sp.diags(dstencil, doffsets, shape=(N, N), format='csc')
        A *= -c * coeff * (1.0 / dx)

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
        Simple linear solver for (I+factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        L = splu(sp.eye(self.params.nvars, format='csc') - factor * self.A)
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
        if self.params.freq >= 0:
            xvalues = np.array([i * self.dx for i in range(self.params.nvars)])
            me[:] = np.sin(np.pi * self.params.freq * (xvalues - self.params.c * t))
        else:
            np.random.seed(1)
            me[:] = np.random.rand(self.params.nvars)
        return me
