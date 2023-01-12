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

        self.xv = np.meshgrid(*[xvalues for _ in range(ndim)])
        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store relevant attributes
        self.c, self.freq, self.dx = c, freq, dx
        self.stencil_type, self.order = stencil_type, order
        self.lintol, self.liniter = lintol, liniter
        self.direct_solver, self.bc = direct_solver, bc

        # register parameters
        self._register('nvars', 'c', 'freq', 'stencil_type', 'order', 'lintol', 'liniter', 'direct_solver', 'bc')

    @property
    def ndim(self):
        return len(self.xv)

    @property
    def nvars(self):
        return self.xv[0].shape

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

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        if self.params.ndim == 1:
            if self.params.freq[0] >= 0:
                me[:] = np.sin(np.pi * self.params.freq[0] * (self.xv[0] - self.params.c * t))
            elif self.params.freq[0] == -1:
                me[:] = np.exp(-0.5 * (((self.xv[0] - (self.params.c * t)) % 1.0 - 0.5) / self.params.sigma) ** 2)

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
