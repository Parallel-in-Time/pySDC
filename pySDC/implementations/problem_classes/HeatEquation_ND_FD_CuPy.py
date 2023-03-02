import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve, cg  # , gmres, minres

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh


# noinspection PyUnusedLocal
class heatNd_forced(ptype):
    """
    Example implementing the ND heat equation with periodic or Diriclet-Zero BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: FD discretization of the ND laplace operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
    """

    dtype_u = cupy_mesh
    dtype_f = imex_cupy_mesh

    def __init__(
        self, nvars, nu, freq, bc, order=2, stencil_type='center', lintol=1e-12, liniter=10000, solver_type='direct'
    ):
        """Initialization routine"""

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
        super().__init__(init=(nvars[0] if ndim == 1 else nvars, None, cp.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'freq',
            'bc',
            'order',
            'stencil_type',
            'lintol',
            'liniter',
            'solver_type',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx (equal in both dimensions) and get discretization matrix A
        if self.bc == 'periodic':
            self.dx = 1.0 / self.nvars[0]
            xvalues = cp.array([i * self.dx for i in range(self.nvars[0])])
        elif self.bc == 'dirichlet-zero':
            self.dx = 1.0 / (self.nvars[0] + 1)
            xvalues = cp.array([(i + 1) * self.dx for i in range(self.nvars[0])])
        else:
            raise ProblemError(f'Boundary conditions {self.bc} not implemented.')

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type=self.stencil_type,
            dx=self.dx,
            size=self.nvars[0],
            dim=self.ndim,
            bc=self.bc,
            cupy=True,
        )
        self.A *= self.nu

        self.xv = cp.meshgrid(*[xvalues for _ in range(self.ndim)])
        self.Id = csp.eye(np.prod(self.nvars), format='csc')

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return len(self.nvars)

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
        f.impl[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        if self.ndim == 1:
            f.expl[:] = cp.sin(np.pi * self.freq[0] * self.xv[0]) * (
                self.nu * np.pi**2 * sum([freq**2 for freq in self.freq]) * cp.cos(t) - cp.sin(t)
            )
        elif self.ndim == 2:
            f.expl[:] = (
                cp.sin(np.pi * self.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.freq[1] * self.xv[1])
                * (self.nu * np.pi**2 * sum([freq**2 for freq in self.freq]) * cp.cos(t) - cp.sin(t))
            )
        elif self.ndim == 3:
            f.expl[:] = (
                cp.sin(np.pi * self.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.freq[2] * self.xv[2])
                * (self.nu * np.pi**2 * sum([freq**2 for freq in self.freq]) * cp.cos(t) - cp.sin(t))
            )

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

        if self.solver_type == 'direct':
            me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.nvars)
        elif self.solver_type == 'CG':
            me[:] = cg(
                self.Id - factor * self.A,
                rhs.flatten(),
                x0=u0.flatten(),
                tol=self.lintol,
                maxiter=self.liniter,
                atol=0,
            )[0].reshape(self.nvars)
        else:
            raise NotImplementedError(f'Solver {self.solver_type} not implemented in GPU heat equation!')
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

        if self.ndim == 1:
            me[:] = cp.sin(np.pi * self.freq[0] * self.xv[0]) * cp.cos(t)
        elif self.ndim == 2:
            me[:] = cp.sin(np.pi * self.freq[0] * self.xv[0]) * cp.sin(np.pi * self.freq[1] * self.xv[1]) * cp.cos(t)
        elif self.ndim == 3:
            me[:] = (
                cp.sin(np.pi * self.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.freq[2] * self.xv[2])
                * cp.cos(t)
            )
        return me


class heatNd_unforced(heatNd_forced):
    dtype_f = cupy_mesh

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

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)

        if self.ndim == 1:
            rho = (2.0 - 2.0 * cp.cos(np.pi * self.freq[0] * self.dx)) / self.dx**2
            me[:] = cp.sin(np.pi * self.freq[0] * self.xv[0]) * cp.exp(-t * self.nu * rho)
        elif self.ndim == 2:
            rho = (2.0 - 2.0 * cp.cos(np.pi * self.freq[0] * self.dx)) / self.dx**2 + (
                2.0 - 2.0 * cp.cos(np.pi * self.freq[1] * self.dx)
            ) / self.dx**2

            me[:] = (
                cp.sin(np.pi * self.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.freq[1] * self.xv[1])
                * cp.exp(-t * self.nu * rho)
            )
        elif self.ndim == 3:
            rho = (
                (2.0 - 2.0 * cp.cos(np.pi * self.freq[0] * self.dx)) / self.dx**2
                + (2.0 - 2.0 * cp.cos(np.pi * self.freq[1] * self.dx))
                + (2.0 - 2.0 * cp.cos(np.pi * self.freq[2] * self.dx)) / self.dx**2
            )
            me[:] = (
                cp.sin(np.pi * self.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.freq[2] * self.xv[2])
                * cp.exp(-t * self.nu * rho)
            )

        return me
