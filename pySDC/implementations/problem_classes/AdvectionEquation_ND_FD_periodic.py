import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class advectionNd_periodic(ptype):
    """
    Example implementing the unforced ND advection equation with periodic BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: FD discretization of the ND grad operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
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

        if 'order' not in problem_params:
            problem_params['order'] = 2
        if 'lintol' not in problem_params:
            problem_params['lintol'] = 1e-12
        if 'liniter' not in problem_params:
            problem_params['liniter'] = 10000
        if 'direct_solver' not in problem_params:
            problem_params['direct_solver'] = False

        essential_keys = ['nvars', 'c', 'freq', 'type', 'order', 'ndim', 'lintol', 'liniter', 'direct_solver']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if problem_params['ndim'] > 3:
            raise ProblemError(f'can work with up to three dimensions, got {problem_params["ndim"]}')
        if type(problem_params['freq']) is not tuple or len(problem_params['freq']) != problem_params['ndim']:
            raise ProblemError(f'need {problem_params["ndim"]} frequencies, got {problem_params["freq"]}')
        for freq in problem_params['freq']:
            if freq % 2 != 0:
                raise ProblemError('need even number of frequencies due to periodic BCs')
        if type(problem_params['nvars']) is not tuple or len(problem_params['nvars']) != problem_params['ndim']:
            raise ProblemError(f'need {problem_params["ndim"]} nvars, got {problem_params["nvars"]}')
        for nvars in problem_params['nvars']:
            if nvars % 2 != 0:
                raise ProblemError('the setup requires nvars = 2^p per dimension')
        if problem_params['nvars'][1:] != problem_params['nvars'][:-1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advectionNd_periodic, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        # compute dx (equal in both dimensions) and get discretization matrix A
        self.dx = 1.0 / self.params.nvars[0]
        self.A = self.__get_A(
            self.params.nvars, self.params.c, self.dx, self.params.ndim, self.params.type, self.params.order
        )
        xvalues = np.array([i * self.dx for i in range(self.params.nvars[0])])
        self.xv = np.meshgrid(*[xvalues for _ in range(self.params.ndim)])
        self.Id = sp.eye(np.prod(self.params.nvars), format='csc')

    @staticmethod
    def __get_A(N, c, dx, ndim, type, order):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes
            type (str): disctretization type
            ndim (int): number of dimensions

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
                [N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))],
            )
        )
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = coeff * sp.diags(dstencil, doffsets, shape=(N[0], N[0]), format='csc')

        if ndim == 2:
            A = sp.kron(A, sp.eye(N[0])) + sp.kron(sp.eye(N[1]), A)
        elif ndim == 3:
            A = (
                sp.kron(A, sp.eye(N[1] * N[0]))
                + sp.kron(sp.eye(N[2] * N[1]), A)
                + sp.kron(sp.kron(sp.eye(N[2]), A), sp.eye(N[0]))
            )
        A *= -c * (1.0 / dx)

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

        if self.params.direct_solver:
            me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.params.nvars)
        else:
            me[:] = gmres(
                self.Id - factor * self.A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=self.params.lintol,
                maxiter=self.params.liniter,
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
        if self.params.ndim == 1:
            me[:] = np.sin(np.pi * self.params.freq[0] * (self.xv[0] - self.params.c * t))
        elif self.params.ndim == 2:
            me[:] = np.sin(np.pi * self.params.freq[0] * (self.xv[0] - self.params.c * t)) * np.sin(
                np.pi * self.params.freq[1] * (self.xv[1] - self.params.c * t)
            )
        elif self.params.ndim == 3:
            me[:] = (
                np.sin(np.pi * self.params.freq[0] * (self.xv[0] - self.params.c * t))
                * np.sin(np.pi * self.params.freq[1] * (self.xv[1] - self.params.c * t))
                * np.sin(np.pi * self.params.freq[2] * (self.xv[2] - self.params.c * t))
            )
        return me
