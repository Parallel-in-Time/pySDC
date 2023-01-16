import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class advectionNd(ptype):
    r"""
    Example implementing the unforced ND advection equation with periodic
    or Dirichlet boundary conditions in :math:`[0,1]^N`,
    and initial solution of the form

    .. math::
        u({\bf x},0) = \prod_{i=1}^N \sin(f\pi x_i),

    with :math:`x_i` the coordinate in :math:`i^{th}` dimension.
    Discretization uses central finite differences.

    Parameters
    ----------
    nvars : int of tuple, optional
        Spatial resolution (same in all dimensions). Using a tuple allows to
        consider several dimensions, e.g nvars=(16,16) for a 2D problem.
    c : float, optional
        Advection speed (same in all dimensions).
    freq : int of tuple, optional
        Spatial frequency :math:`f` of the initial conditions, can be tuple.
    stencil_type : str, optional
        Type of the finite difference stencil.
    order : int, optional
        Order of the finite difference discretization.
    lintol : float, optional
        Tolerance for spatial solver (GMRES).
    liniter : int, optional
        Max. iterations number for GMRES.
    direct_solver : bool, optional
        Whether to solve directly or use GMRES.
    bc : str, optional
        Boundary conditions, either "periodic" or "dirichlet".
    sigma : float, optional
        If freq=-1 and ndim=1, uses a Gaussian initial solution of the form

    .. math::
        u(x,0) = e^{
            \frac{\displaystyle 1}{\displaystyle 2}
            \left(
                \frac{\displaystyle x-1/2}{\displaystyle \sigma}
            \right)^2
            }

    Attributes
    ----------
    A: sparse matrix (CSC)
        FD discretization matrix of the ND grad operator.
    ndim: int
        Number of space dimensions.
    dx: float
        Distance between two spatial nodes (here: being the same in all dimensions).
    Id: sparse matrix (CSC)
        Identity matrix of the same dimension as A

    Notes
    -----
    Args can be set as values or as tuples, which will increase the dimension.
    Do, however, take care that all spatial parameters have the same dimension.
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
        self.nvars, self.c = nvars, c
        self.stencil_type, self.order, self.bc = stencil_type, order, bc
        self.freq, self.sigma = freq, sigma
        self.lintol, self.liniter, self.direct_solver = lintol, liniter, direct_solver

        # register parameters
        self._register('nvars', 'c', 'stencil_type', 'order', 'bc', readOnly=True)
        self._register('freq', 'sigma', 'lintol', 'liniter', 'direct_solver')

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return len(self.nvars)

    @property
    def dx(self):
        """Size of the mesh (in all dimensions)"""
        return self.xvalues[1] - self.xvalues[0]

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs.

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
        sol : dtype_u
            The solution of the linear solver.
        """
        direct_solver, Id, A, nvars, lintol, liniter, sol = (
            self.direct_solver,
            self.Id,
            self.A,
            self.nvars,
            self.lintol,
            self.liniter,
            self.u_init,
        )

        if direct_solver:
            sol[:] = spsolve(Id - factor * A, rhs.flatten()).reshape(nvars)
        else:
            sol[:] = gmres(Id - factor * A, rhs.flatten(), x0=u0.flatten(), tol=lintol, maxiter=liniter, atol=0,)[
                0
            ].reshape(nvars)

        return sol

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t

        Parameters
        ----------
        t : float
            Time of the exact solution.
        **kwargs : dict
            Additional arguments (that won't be used).

        Returns
        -------
        sol : dtype_u
            The exact solution.
        """
        # Initialize pointers and variables
        ndim, freq, x, c, sigma, sol = self.ndim, self.freq, self.xvalues, self.c, self.sigma, self.u_init

        if ndim == 1:
            if freq[0] >= 0:
                sol[:] = np.sin(np.pi * freq[0] * (x - c * t))
            elif freq[0] == -1:
                # Gaussian initial solution
                sol[:] = np.exp(-0.5 * (((x - (c * t)) % 1.0 - 0.5) / sigma) ** 2)

        elif ndim == 2:
            sol[:] = np.sin(np.pi * freq[0] * (x[None, :] - c * t)) * np.sin(np.pi * freq[1] * (x[:, None] - c * t))

        elif ndim == 3:
            sol[:] = (
                np.sin(np.pi * freq[0] * (x[None, :, None] - c * t))
                * np.sin(np.pi * freq[1] * (x[:, None, None] - c * t))
                * np.sin(np.pi * freq[2] * (x[None, None, :] - c * t))
            )

        return sol
