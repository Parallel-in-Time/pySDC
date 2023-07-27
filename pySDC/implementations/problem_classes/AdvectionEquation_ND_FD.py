import numpy as np

from pySDC.implementations.problem_classes.generic_ND_FD import GenericNDimFinDiff


# noinspection PyUnusedLocal
class advectionNd(GenericNDimFinDiff):
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
    solver_type : str, optional
        Solve the linear system directly or using GMRES or CG
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
    Id: sparse matrix (CSC)
        Identity matrix of the same dimension as A

    Note
    ----
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
        solver_type='direct',
        bc='periodic',
        sigma=6e-2,
    ):
        super().__init__(nvars, -c, 1, freq, stencil_type, order, lintol, liniter, solver_type, bc)

        if solver_type == 'CG':  # pragma: no cover
            self.logger.warn('CG is not usually used for advection equation')
        self._makeAttributeAndRegister('c', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('sigma', localVars=locals())

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
        if 'u_init' in kwargs.keys() or 't_init' in kwargs.keys():
            self.logger.warn(
                f'{type(self).__name__} uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
            )

        # Initialize pointers and variables
        ndim, freq, c, sigma, sol = self.ndim, self.freq, self.c, self.sigma, self.u_init

        if ndim == 1:
            x = self.grids
            if freq[0] >= 0:
                sol[:] = np.sin(np.pi * freq[0] * (x - c * t))
            elif freq[0] == -1:
                # Gaussian initial solution
                sol[:] = np.exp(-0.5 * (((x - (c * t)) % 1.0 - 0.5) / sigma) ** 2)

        elif ndim == 2:
            x, y = self.grids
            sol[:] = np.sin(np.pi * freq[0] * (x - c * t)) * np.sin(np.pi * freq[1] * (y - c * t))

        elif ndim == 3:
            x, y, z = self.grids
            sol[:] = (
                np.sin(np.pi * freq[0] * (x - c * t))
                * np.sin(np.pi * freq[1] * (y - c * t))
                * np.sin(np.pi * freq[2] * (z - c * t))
            )

        return sol
