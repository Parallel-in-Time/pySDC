import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype, get_finite_difference_stencil
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class heat2d_periodic(ptype):
    """
    Example implementing the unforced 2D heat equation with periodic BCs in [0,1]^2,
    discretized using central finite differences

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (here: being the same in both dimensions)
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
        essential_keys = ['nvars', 'nu', 'freq']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if problem_params['freq'] % 2 != 0:
            raise ProblemError('need even number of frequencies due to periodic BCs')
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] != problem_params['nvars'][1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')
        if 'order' not in problem_params:
            problem_params['order'] = 2

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_periodic, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        # compute dx (equal in both dimensions) and get discretization matrix A
        self.dx = 1.0 / self.params.nvars[0]
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx, self.params.order)

    @staticmethod
    def __get_A(N, nu, dx, order):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil, zero_pos, _ = get_finite_difference_stencil(derivative=2, order=order, type='center')
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(
            (
                [N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))],
            )
        )
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = sp.diags(dstencil, doffsets, shape=(N[0], N[0]), format='csc')

        A = sp.kron(A, sp.eye(N[0])) + sp.kron(sp.eye(N[1]), A)
        A *= nu / (dx**2)

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
        f[:] = self.A.dot(u.flatten()).reshape(self.params.nvars)
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
        me[:] = cg(
            sp.eye(self.params.nvars[0] * self.params.nvars[1], format='csc') - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            tol=1e-12,
            atol=0,
        )[0].reshape(self.params.nvars)
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
        xvalues = np.array([i * self.dx for i in range(self.params.nvars[0])])
        xv, yv = np.meshgrid(xvalues, xvalues)
        me[:] = (
            np.sin(np.pi * self.params.freq * xv)
            * np.sin(np.pi * self.params.freq * yv)
            * np.exp(-t * self.params.nu * 2 * (np.pi * self.params.freq) ** 2)
        )
        return me
