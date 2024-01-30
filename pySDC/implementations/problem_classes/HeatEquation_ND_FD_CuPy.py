import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve, cg  # , gmres, minres

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh


class heatNd_forced(ptype):  # pragma: no cover
    r"""
    This class implements the unforced :math:`N`-dimensional heat equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \nu
        \left(
            \frac{\partial^2 u}{\partial x^2_1} + .. + \frac{\partial^2 u}{\partial x^2_N}
        \right)

    for :math:`(x_1,..,x_N) \in [0, 1]^{N}` with :math:`N \leq 3`. The initial solution is of the form

    .. math::
        u({\bf x},0) = \prod_{i=1}^N \sin(\pi k_i x_i).

    The spatial term is discretized using finite differences.

    This class uses the ``CuPy`` package in order to make ``pySDC`` available for GPUs.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution (same in all dimensions). Using a tuple allows to
        consider several dimensions, e.g ``nvars=(16,16)`` for a 2D problem.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    freq : int, optional
        Spatial frequency :math:`k_i` of the initial conditions, can be tuple.
    stencil_type : str, optional
        Type of the finite difference stencil.
    order : int, optional
        Order of the finite difference discretization.
    lintol : float, optional
        Tolerance for spatial solver.
    liniter : int, optional
        Max. iterations number for spatial solver.
    solver_type : str, optional
        Solve the linear system directly or using CG.
    bc : str, optional
        Boundary conditions, either ``'periodic'`` or ``'dirichlet'``.
    sigma : float, optional
        If ``freq=-1`` and ``ndim=1``, uses a Gaussian initial solution of the form

        .. math::
            u(x,0) = e^{
                \frac{\displaystyle 1}{\displaystyle 2}
                \left(
                    \frac{\displaystyle x-1/2}{\displaystyle \sigma}
                \right)^2
                }

    Attributes
    ----------
    A : sparse matrix (CSC)
        FD discretization matrix of the ND grad operator.
    Id : sparse matrix (CSC)
        Identity matrix of the same dimension as A
    """

    dtype_u = cupy_mesh
    dtype_f = imex_cupy_mesh

    def __init__(
        self,
        nvars=512,
        nu=0.1,
        freq=2,
        bc='periodic',
        order=2,
        stencil_type='center',
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
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

        self.A, _ = problem_helper.get_finite_difference_matrix(
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
            Solution.
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
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
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
    r"""
    This class implements the forced :math:`N`-dimensional heat equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \nu
        \left(
            \frac{\partial^2 u}{\partial x^2_1} + .. + \frac{\partial^2 u}{\partial x^2_N}
        \right) + f({\bf x}, t)

    for :math:`(x_1,..,x_N) \in [0, 1]^{N}` with :math:`N \leq 3`, and forcing term

    .. math::
        f({\bf x}, t) = \prod_{i=1}^N \sin(\pi k_i x_i) \left(
            \nu \pi^2 \sum_{i=1}^N k_i^2 \cos(t) - \sin(t)
        \right),

    where :math:`k_i` denotes the frequency in the :math:`i^{th}` dimension. The exact solution is

    .. math::
        u({\bf x}, t) = \prod_{i=1}^N \sin(\pi k_i x_i) \cos(t).

    The spatial term is discretized using finite differences.

    The implementation is this class uses the ``CuPy`` package in order to make ``pySDC`` available for GPUs.
    """

    dtype_f = cupy_mesh

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
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)

        return f

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
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
