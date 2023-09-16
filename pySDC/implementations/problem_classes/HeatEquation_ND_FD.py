import numpy as np

from pySDC.implementations.problem_classes.generic_ND_FD import GenericNDimFinDiff
from pySDC.implementations.datatype_classes.mesh import imex_mesh


class heatNd_unforced(GenericNDimFinDiff):
    r"""
    This class implements the unforced N-dimensional heat equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \nu
        \left(
            \frac{\partial^2 u}{\partial x^2_1} + .. + \frac{\partial^2 u}{\partial x^2_N}
        \right)

    for :math:`(x_1,..,x_N) \in [0, 1]^{N}` with :math:`N \leq 3`. The initial solution is of the form

    .. math::
        u({\bf x},0) = \prod_{i=1}^N \sin(\pi k_i x_i).

    The spatial term is discretized using finite differences.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution (same in all dimensions). Using a tuple allows to
        consider several dimensions, e.g nvars=(16,16) for a 2D problem.
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
    A : sparse matrix (CSC)
        FD discretization matrix of the ND operator.
    Id : sparse matrix (CSC)
        Identity matrix of the same dimension as A
    """

    def __init__(
        self,
        nvars=512,
        nu=0.1,
        freq=2,
        stencil_type='center',
        order=2,
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
        bc='periodic',
        sigma=6e-2,
    ):
        """Initialization routine"""
        super().__init__(nvars, nu, 2, freq, stencil_type, order, lintol, liniter, solver_type, bc)
        if solver_type == 'GMRES':
            self.logger.warning('GMRES is not usually used for heat equation')
        self._makeAttributeAndRegister('nu', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('sigma', localVars=locals())

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        sol : dtype_u
            The exact solution.
        """
        if 'u_init' in kwargs.keys() or 't_init' in kwargs.keys():
            self.logger.warning(
                f'{type(self).__name__} uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
            )

        ndim, freq, nu, sigma, dx, sol = self.ndim, self.freq, self.nu, self.sigma, self.dx, self.u_init

        if ndim == 1:
            x = self.grids
            rho = (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2
            if freq[0] > 0:
                sol[:] = np.sin(np.pi * freq[0] * x) * np.exp(-t * nu * rho)
            elif freq[0] == -1:  # Gaussian
                sol[:] = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2) * np.exp(-t * nu * rho)
        elif ndim == 2:
            rho = (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2 + (
                2.0 - 2.0 * np.cos(np.pi * freq[1] * dx)
            ) / dx**2
            x, y = self.grids
            sol[:] = np.sin(np.pi * freq[0] * x) * np.sin(np.pi * freq[1] * y) * np.exp(-t * nu * rho)
        elif ndim == 3:
            rho = (
                (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2
                + (2.0 - 2.0 * np.cos(np.pi * freq[1] * dx))
                + (2.0 - 2.0 * np.cos(np.pi * freq[2] * dx)) / dx**2
            )
            x, y, z = self.grids
            sol[:] = (
                np.sin(np.pi * freq[0] * x)
                * np.sin(np.pi * freq[1] * y)
                * np.sin(np.pi * freq[2] * z)
                * np.exp(-t * nu * rho)
            )

        return sol


class heatNd_forced(heatNd_unforced):
    r"""
    This class implements the forced N-dimensional heat equation with periodic boundary conditions

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

    The spatial term is discretized using central finite differences.
    """

    dtype_f = imex_mesh

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

        f = self.f_init
        f.impl[:] = self.A.dot(u.flatten()).reshape(self.nvars)

        ndim, freq, nu = self.ndim, self.freq, self.nu
        if ndim == 1:
            x = self.grids
            f.expl[:] = np.sin(np.pi * freq[0] * x) * (
                nu * np.pi**2 * sum([freq**2 for freq in freq]) * np.cos(t) - np.sin(t)
            )
        elif ndim == 2:
            x, y = self.grids
            f.expl[:] = (
                np.sin(np.pi * freq[0] * x)
                * np.sin(np.pi * freq[1] * y)
                * (nu * np.pi**2 * sum([freq**2 for freq in freq]) * np.cos(t) - np.sin(t))
            )
        elif ndim == 3:
            x, y, z = self.grids
            f.expl[:] = (
                np.sin(np.pi * freq[0] * x)
                * np.sin(np.pi * freq[1] * y)
                * np.sin(np.pi * freq[2] * z)
                * (nu * np.pi**2 * sum([freq**2 for freq in freq]) * np.cos(t) - np.sin(t))
            )

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        sol : dtype_u
            The exact solution.
        """
        ndim, freq, sol = self.ndim, self.freq, self.u_init
        if ndim == 1:
            x = self.grids
            sol[:] = np.sin(np.pi * freq[0] * x) * np.cos(t)
        elif ndim == 2:
            x, y = self.grids
            sol[:] = np.sin(np.pi * freq[0] * x) * np.sin(np.pi * freq[1] * y) * np.cos(t)
        elif ndim == 3:
            x, y, z = self.grids
            sol[:] = np.sin(np.pi * freq[0] * x) * np.sin(np.pi * freq[1] * y) * np.sin(np.pi * freq[2] * z) * np.cos(t)
        return sol
