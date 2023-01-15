import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class advectionNd(ptype):
    """
    Example implementing the unforced ND advection equation with periodic BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: FD discretization of the ND grad operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
    """

    def __init__(
        self,
        nvars=512,
        c=1.0,
        freq=2,
        stencil_type='center',
        order=2,
        lintol=1e-12,
        liniter=10000,
        direct_solver=True,
        bc='periodic',
        sigma=6e-2,
    ):
        """
        Initialization routine

        Args can be set as values or as tuples, which will increase the dimension.
        Do, however, take care that all spatial parameters have the same dimension.

        Args:
            nvars (int): Spatial resolution, can be tuple
            c (float): Advection speed, can be tuple
            freq (int): Spatial frequency of the initial conditions, can be tuple
            stencil_type (str): Type of the finite difference stencil
            order (int): Order of the finite difference discretization
            lintol (float): Tolerance for spatial solver (GMRES)
            liniter (int): Max. iterations for GMRES
            direct_solver (bool): Whether to solve directly or use GMRES
            bc (str): Boundary conditions
        """

        # make sure parameters have the correct types
        if not type(nvars) in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')
        if not type(freq) in [int, tuple]:
            raise ProblemError('freq should be either tuple or int')

        # transforms nvars into a tuple
        if type(nvars) is int:
            nvars = (nvars,)

        # automatically determine ndim from nvars
        ndim = len(nvars)
        if ndim > 3:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

        # eventually extend freq to other dimension
        if type(freq) is int:
            freq = (freq,) * ndim
        if len(freq) != ndim:
            raise ProblemError(f'len(freq)={len(freq)}, different to ndim={ndim}')

        # check values for freq and nvars
        for f in freq:
            if ndim == 1 and f == -1:
                # use Gaussian initial solution in 1D
                bc = 'periodic'
                break
            if f % 2 != 0 and bc == 'periodic':
                raise ProblemError('need even number of frequencies due to periodic BCs')
        for nvar in nvars:
            if nvar % 2 != 0 and bc == 'periodic':
                raise ProblemError('the setup requires nvars = 2^p per dimension')
            if (nvar + 1) % 2 != 0 and bc == 'dirichlet-zero':
                raise ProblemError('setup requires nvars = 2^p - 1')
        if ndim > 1 and nvars[1:] != nvars[:-1]:
            raise ProblemError('need a square domain, got %s' % nvars)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(
            init=(nvars, None, np.dtype('float64')),
            dtype_u=mesh,
            dtype_f=mesh,
        )

        # compute dx (equal in both dimensions) and get discretization matrix A
        if bc == 'periodic':
            xvalues = np.linspace(0, 1, num=nvars[0], endpoint=False)
        elif bc == 'dirichlet-zero':
            xvalues = np.linspace(0, 1, num=nvars[0] + 2)[1:-1]
        else:
            raise ProblemError(f'Boundary conditions {self.params.bc} not implemented.')
        dx = xvalues[1] - xvalues[0]

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=1,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars[0],
            dim=ndim,
            bc=bc,
        )
        self.A *= -c

        self.xvalues = xvalues
        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store relevant attributes
        self.nvars, self.ndim, self.c = nvars, ndim, c
        self.stencil_type, self.order, self.bc = stencil_type, order, bc
        self.freq, self.sigma = freq, sigma
        self.lintol, self.liniter, self.direct_solver = lintol, liniter, direct_solver

        # register parameters
        self._register('nvars', 'ndim', 'c', 'stencil_type', 'order', 'bc', readOnly=True)
        self._register('freq', 'gamma', 'lintol', 'liniter', 'direct_solver')

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
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
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
        direct_solver, Id, A, nvars, lintol, liniter = (
            self.direct_solver,
            self.Id,
            self.A,
            self.nvars,
            self.lintol,
            self.liniter,
        )
        me = self.dtype_u(self.init)

        if direct_solver:
            me[:] = spsolve(Id - factor * A, rhs.flatten()).reshape(nvars)
        else:
            me[:] = gmres(Id - factor * A, rhs.flatten(), x0=u0.flatten(), tol=lintol, maxiter=liniter, atol=0,)[
                0
            ].reshape(nvars)

        return me

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            me: exact solution
        """
        # Initialize pointers and variables
        ndim, freq, x, c, sigma = self.ndim, self.freq, self.xvalues, self.c, self.sigma
        me = self.dtype_u(self.init)

        if ndim == 1:
            if freq[0] >= 0:
                me[:] = np.sin(np.pi * freq[0] * (x - c * t))
            elif freq[0] == -1:
                # Gaussian initial solution
                me[:] = np.exp(-0.5 * (((x - (c * t)) % 1.0 - 0.5) / sigma) ** 2)

        elif ndim == 2:
            me[:] = np.sin(np.pi * freq[0] * (x[None, :] - c * t)) * np.sin(np.pi * freq[1] * (x[:, None] - c * t))

        elif ndim == 3:
            me[:] = (
                np.sin(np.pi * freq[0] * (x[None, :, None] - c * t))
                * np.sin(np.pi * freq[1] * (x[:, None, None] - c * t))
                * np.sin(np.pi * freq[2] * (x[None, None, :] - c * t))
            )

        return me
