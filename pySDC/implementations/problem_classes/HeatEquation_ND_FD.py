import numpy as np

from pySDC.implementations.problem_classes.generic_ND_FD import GenericNDimFinDiff
from pySDC.implementations.datatype_classes.mesh import imex_mesh


class heatNd_unforced(GenericNDimFinDiff):
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
        super().__init__(nvars, nu, 2, freq, stencil_type, order, lintol, liniter, solver_type, bc)
        if solver_type == 'GMRES':
            self.logger.warn('GMRES is not usually used for heat equation')
        self._makeAttributeAndRegister('nu', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('sigma', localVars=locals())

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
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
    """
    Example implementing the ND heat equation with periodic or Diriclet-Zero BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: FD discretization of the ND laplace operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
    """

    dtype_f = imex_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
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
