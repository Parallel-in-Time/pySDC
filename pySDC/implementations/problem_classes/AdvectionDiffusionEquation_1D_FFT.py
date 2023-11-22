import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class advectiondiffusion1d_imex(ptype):
    r"""
    Example implementing the unforced one-dimensional advection diffusion equation

    .. math::
        \frac{\partial u(x,t)}{\partial t} = - c \frac{\partial u(x,t)}{\partial x} + \nu \frac{\partial^2 u(x,t)}{\partial x^2}

    with periodic boundary conditions in :math:`[-\frac{L}{2}, \frac{L}{2}]` in spectral space. The advection part
    :math:`- c \frac{\partial u(x,t)}{\partial x}` is treated explicitly, whereas the diffusion part
    :math:`\nu \frac{\partial^2 u(x,t)}{\partial x^2}` will be treated numerically in an implicit way. The exact solution is
    given by

    .. math::
        u(x, t) = \sin(\omega (x - c t)) \exp(-t \nu \omega^2)

    for :math:`\omega=2 \pi k`, where :math:`k` denotes the wave number. Fast Fourier transform is used for the spatial
    discretization.

    Parameters
    ----------
    nvars : int, optional
        Number of points in spatial discretization.
    c : float, optional
        Advection speed :math:`c`.
    freq : int, optional
        Wave number :math:`k`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    L : float, optional
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

    def __init__(self, nvars=256, c=1.0, freq=-1, nu=0.02, L=1.0):
        """Initialization routine"""

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'c', 'freq', 'nu', 'L', localVars=locals(), readOnly=True)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (self.nvars) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        self.xvalues = np.array([i * self.L / self.nvars - self.L / 2.0 for i in range(self.nvars)])

        kx = np.zeros(self.init[0] // 2 + 1)
        for i in range(0, len(kx)):
            kx[i] = 2 * np.pi / self.L * i

        self.ddx = kx * 1j
        self.lap = -(kx**2)

        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        tmp_u = np.fft.rfft(u)
        tmp_impl = self.nu * self.lap * tmp_u
        tmp_expl = -self.c * self.ddx * tmp_u
        f.impl[:] = np.fft.irfft(tmp_impl)
        f.expl[:] = np.fft.irfft(tmp_expl)

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        tmp = np.fft.rfft(rhs) / (1.0 - self.nu * factor * self.lap)
        me[:] = np.fft.irfft(tmp)

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
            The exact solution.
        """

        me = self.dtype_u(self.init, val=0.0)
        if self.freq > 0:
            omega = 2.0 * np.pi * self.freq
            me[:] = np.sin(omega * (self.xvalues - self.c * t)) * np.exp(-t * self.nu * omega**2)
        elif self.freq == 0:
            np.random.seed(1)
            me[:] = np.random.rand(self.nvars)
        else:
            t00 = 0.08
            if self.nu > 0:
                nbox = int(np.ceil(np.sqrt(4.0 * self.nu * (t00 + t) * 37.0 / (self.L**2))))
                for k in range(-nbox, nbox + 1):
                    for i in range(self.init[0]):
                        x = self.xvalues[i] - self.c * t + k * self.L
                        me[i] += np.sqrt(t00) / np.sqrt(t00 + t) * np.exp(-(x**2) / (4.0 * self.nu * (t00 + t)))
            else:
                raise ParameterError("There is no exact solution implemented for negative frequency and negative nu!")
        return me


class advectiondiffusion1d_implicit(advectiondiffusion1d_imex):
    r"""
    Example implementing the unforced one-dimensional advection diffusion equation

    .. math::
        \frac{\partial u(x,t)}{\partial t} = - c \frac{\partial u(x,t)}{\partial x} + \nu \frac{\partial^2 u(x,t)}{\partial x^2}

    with periodic boundary conditions in :math:`[-\frac{L}{2}, \frac{L}{2}]` in spectral space. This class implements the
    problem solving it with fully-implicit time-stepping. The exact solution is given by

    .. math::
        u(x, t) = \sin(\omega (x - c t)) \exp(-t \nu \omega^2)

    for :math:`\omega=2 \pi k`, where :math:`k` denotes the wave number. Fast Fourier transform is used for the spatial
    discretization.

    Note
    ----
    This class has the same attributes as the class it inherits from.
    """

    dtype_u = mesh
    dtype_f = mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        tmp_u = np.fft.rfft(u)
        tmp = self.nu * self.lap * tmp_u - self.c * self.ddx * tmp_u
        f[:] = np.fft.irfft(tmp)

        self.work_counters['rhs']
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion and advection part (both are linear!).

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        tmp = np.fft.rfft(rhs) / (1.0 - factor * (self.nu * self.lap - self.c * self.ddx))
        me[:] = np.fft.irfft(tmp)

        return me
