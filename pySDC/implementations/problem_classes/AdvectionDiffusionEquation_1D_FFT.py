import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class advectiondiffusion1d_imex(ptype):
    r"""
    Example implementing the unforced one-dimensional advection diffusion equation

    .. math::
        \frac{\partial u}{\partial t} = - c \frac{\partial u}{\partial x} + \nu \frac{\partial^2 u}{\partial x^2}

    with periodic BC in [-L/2, L/2] in spectral space. The advection part :math:`- c \frac{\partial u}{\partial x}` is
    treated explicitly, whereas the diffusion part :math:`\nu \frac{\partial^2 u}{\partial x^2}` will be treated numerically
    in an implicit way. The exact solution is given by

    .. math::
        u(x, t) = \sin(\omega (x - c t)) \exp(-t \nu \omega^2)

    for :math:`\omega=2 \pi k`, where :math:`k` denotes the wave number. Fast Fourier transform is used for the spatial
    discretization.

    Parameters
    ----------
    nvars : int
        Number of points in spatial discretization.
    c : float
        Advection speed.
    freq : int
        Wave number :math:`k`.
    nu : float
        Diffusion coefficient :math:`\nu`.
    L : int
        Denotes the period of the function to be approximated for the Fourier transform.

    Attributes
    ----------
    xvalues : np.1darray
        Contains the grid points in space.
    ddx : np.1darray
        Spectral operator for gradient.
    lap : np.1darray
        Spectral operator for Laplacian.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars, c, freq, nu, L=1.0):
        """Initialization routine"""

        # these parameters will be used later, so assert their existence
        #essential_keys = ['nvars', 'c', 'freq', 'nu']
        #for key in essential_keys:
        #    if key not in problem_params:
        #        msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
        #        raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'c', 'freq', 'nu', 'L', localVars=locals(), readOnly=True)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (self.params.nvars) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        self.xvalues = np.array(
            [i * self.params.L / self.params.nvars - self.params.L / 2.0 for i in range(self.params.nvars)]
        )

        kx = np.zeros(self.init[0] // 2 + 1)
        for i in range(0, len(kx)):
            kx[i] = 2 * np.pi / self.params.L * i

        self.ddx = kx * 1j
        self.lap = -(kx**2)

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
        tmp_u = np.fft.rfft(u)
        tmp_impl = self.params.nu * self.lap * tmp_u
        tmp_expl = -self.params.c * self.ddx * tmp_u
        f.impl[:] = np.fft.irfft(tmp_impl)
        f.expl[:] = np.fft.irfft(tmp_expl)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        tmp = np.fft.rfft(rhs) / (1.0 - self.params.nu * factor * self.lap)
        me[:] = np.fft.irfft(tmp)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init, val=0.0)
        if self.params.freq > 0:
            omega = 2.0 * np.pi * self.params.freq
            me[:] = np.sin(omega * (self.xvalues - self.params.c * t)) * np.exp(-t * self.params.nu * omega**2)
        elif self.params.freq == 0:
            np.random.seed(1)
            me[:] = np.random.rand(self.params.nvars)
        else:
            t00 = 0.08
            if self.params.nu > 0:
                nbox = int(np.ceil(np.sqrt(4.0 * self.params.nu * (t00 + t) * 37.0 / (self.params.L**2))))
                for k in range(-nbox, nbox + 1):
                    for i in range(self.init[0]):
                        x = self.xvalues[i] - self.params.c * t + k * self.params.L
                        me[i] += (
                            np.sqrt(t00) / np.sqrt(t00 + t) * np.exp(-(x**2) / (4.0 * self.params.nu * (t00 + t)))
                        )
        return me


class advectiondiffusion1d_implicit(advectiondiffusion1d_imex):
    """
    Example implementing the unforced 1D advection diffusion equation with periodic BC in [-L/2, L/2] in spectral space,
    fully-implicit time-stepping
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type (will be passed to parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advectiondiffusion1d_implicit, self).__init__(
            problem_params=problem_params, dtype_u=dtype_u, dtype_f=dtype_f
        )

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
        tmp_u = np.fft.rfft(u)
        tmp = self.params.nu * self.lap * tmp_u - self.params.c * self.ddx * tmp_u
        f[:] = np.fft.irfft(tmp)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion and advection part (both are linear!)

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        tmp = np.fft.rfft(rhs) / (1.0 - factor * (self.params.nu * self.lap - self.params.c * self.ddx))
        me[:] = np.fft.irfft(tmp)

        return me
