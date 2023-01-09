import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve

from pySDC.core.Errors import ParameterError, ProblemError
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
        ndim=None,
    ):
        """
        Initialization routine

        Args can be set as values or as tuples, which will increase the dimension. Do, however, take care that all
        spatial parameters have the same dimension.

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
            ndim (int): Number of dimensions. Is set automatically if left at None.
        """

        # make sure parameters have the correct form
        if not (type(nvars) is tuple and type(freq) is tuple) and not (type(nvars) is int and type(freq) is int):
            print(nvars, freq)
            raise ProblemError('Type of nvars and freq must be both either int or both tuple')

        if ndim is None:
            if type(nvars) is int:
                ndim = 1
            elif type(nvars) is tuple:
                ndim = len(nvars)

        if ndim > 3:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

        if type(freq) is tuple:
            for f in freq:
                if f % 2 != 0 and bc == 'periodic':
                    raise ProblemError('need even number of frequencies due to periodic BCs')
        else:
            if freq % 2 != 0 and freq != -1 and bc == 'periodic':
                raise ProblemError('need even number of frequencies due to periodic BCs')

        if type(nvars) is tuple:
            for nvar in nvars:
                if nvar % 2 != 0 and bc == 'periodic':
                    raise ProblemError('the setup requires nvars = 2^p per dimension')
                if (nvar + 1) % 2 != 0 and bc == 'dirichlet-zero':
                    raise ProblemError('setup requires nvars = 2^p - 1')
            if nvars[1:] != nvars[:-1]:
                raise ProblemError('need a square domain, got %s' % nvars)
        else:
            if nvars % 2 != 0 and bc == 'periodic':
                raise ProblemError('the setup requires nvars = 2^p per dimension')
            if (nvars + 1) % 2 != 0 and bc == 'dirichlet-zero':
                raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advectionNd, self).__init__(
            init=(nvars, None, np.dtype('float64')),
            dtype_u=mesh,
            dtype_f=mesh,
        )

        # store parameters
        self.nvars = nvars
        self.c = c
        self.freq = freq
        self.stencil_type = stencil_type
        self.order = order
        self.lintol = lintol
        self.liniter = liniter
        self.direct_solver = direct_solver
        self.bc = bc
        self.ndim = ndim

        if self.ndim == 1:
            if type(self.nvars) is not tuple:
                self.nvars = (self.nvars,)
            if type(self.freq) is not tuple:
                self.freq = (self.freq,)

        # compute dx (equal in both dimensions) and get discretization matrix A
        if self.bc == 'periodic':
            self.dx = 1.0 / self.nvars[0]
            xvalues = np.array([i * self.dx for i in range(self.nvars[0])])
        elif self.bc == 'dirichlet-zero':
            self.dx = 1.0 / (self.nvars[0] + 1)
            xvalues = np.array([(i + 1) * self.dx for i in range(self.nvars[0])])
        else:
            raise ProblemError(f'Boundary conditions {self.bc} not implemented.')

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=1,
            order=self.order,
            stencil_type=self.stencil_type,
            dx=self.dx,
            size=self.nvars[0],
            dim=self.ndim,
            bc=self.bc,
        )
        self.A *= -self.c

        self.xv = np.meshgrid(*[xvalues for _ in range(self.ndim)])
        self.Id = sp.eye(np.prod(self.nvars), format='csc')

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

        me = self.dtype_u(self.init)

        if self.direct_solver:
            me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.nvars)
        else:
            me[:] = gmres(
                self.Id - factor * self.A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=self.lintol,
                maxiter=self.liniter,
                atol=0,
            )[0].reshape(self.nvars)
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
        if self.ndim == 1:
            if self.freq[0] >= 0:
                me[:] = np.sin(np.pi * self.freq[0] * (self.xv[0] - self.c * t))
            elif self.freq[0] == -1:
                me[:] = np.exp(-0.5 * (((self.xv[0] - (self.c * t)) % 1.0 - 0.5) / self.sigma) ** 2)

        elif self.ndim == 2:
            me[:] = np.sin(np.pi * self.freq[0] * (self.xv[0] - self.c * t)) * np.sin(
                np.pi * self.freq[1] * (self.xv[1] - self.c * t)
            )
        elif self.ndim == 3:
            me[:] = (
                np.sin(np.pi * self.freq[0] * (self.xv[0] - self.c * t))
                * np.sin(np.pi * self.freq[1] * (self.xv[1] - self.c * t))
                * np.sin(np.pi * self.freq[2] * (self.xv[2] - self.c * t))
            )
        return me
