from __future__ import division
import numpy as np
import pyfftw

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class advectiondiffusion1d_imex(ptype):
    """
    Example implementing the unforced 1D advection diffusion equation with periodic BC in [-L/2, L/2] in spectral space,
    IMEX time-stepping

    Attributes:
        xvalues: grid points in space
        ddx: spectral operator for gradient
        lap: spectral operator for Laplacian
        rfft_object: planned real-valued FFT for forward transformation
        irfft_object: planned IFFT for backward transformation, real-valued output
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'c', 'freq', 'nu', 'L']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advectiondiffusion1d_imex, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                                        params=problem_params)

        self.xvalues = np.array([i * self.params.L / self.params.nvars - self.params.L / 2.0
                                 for i in range(self.params.nvars)])

        kx = np.zeros(self.init // 2 + 1)
        for i in range(0, len(kx)):
            kx[i] = 2 * np.pi / self.params.L * i

        self.ddx = kx * 1j
        self.lap = -kx ** 2

        rfft_in = pyfftw.empty_aligned(self.init, dtype='float64')
        fft_out = pyfftw.empty_aligned(self.init // 2 + 1, dtype='complex128')
        ifft_in = pyfftw.empty_aligned(self.init // 2 + 1, dtype='complex128')
        irfft_out = pyfftw.empty_aligned(self.init, dtype='float64')
        self.rfft_object = pyfftw.FFTW(rfft_in, fft_out, direction='FFTW_FORWARD')
        self.irfft_object = pyfftw.FFTW(ifft_in, irfft_out, direction='FFTW_BACKWARD')

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
        tmp_u = self.rfft_object(u.values)
        tmp_impl = self.params.nu * self.lap * tmp_u
        tmp_expl = -self.params.c * self.ddx * tmp_u
        f.impl.values[:] = self.irfft_object(tmp_impl)
        f.expl.values[:] = self.irfft_object(tmp_expl)

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
        tmp = self.rfft_object(rhs.values) / (1.0 - self.params.nu * factor * self.lap)
        me.values[:] = self.irfft_object(tmp)

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
            me.values = np.sin(omega * (self.xvalues - self.params.c * t)) * np.exp(-t * self.params.nu * omega ** 2)
        elif self.params.freq == 0:
            np.random.seed(1)
            me.values = np.random.rand(self.params.nvars)
        else:
            t00 = 0.08
            if self.params.nu > 0:
                nbox = int(np.ceil(np.sqrt(4.0 * self.params.nu * (t00 + t) * 37.0 / (self.params.L ** 2))))
                for k in range(-nbox, nbox + 1):
                    for i in range(self.init):
                        x = self.xvalues[i] - self.params.c * t + k * self.params.L
                        me.values[i] += np.sqrt(t00) / np.sqrt(t00 + t) * \
                            np.exp(-x ** 2 / (4.0 * self.params.nu * (t00 + t)))
        return me


class advectiondiffusion1d_implicit(advectiondiffusion1d_imex):
    """
    Example implementing the unforced 1D advection diffusion equation with periodic BC in [-L/2, L/2] in spectral space,
    fully-implicit time-stepping
    """

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
        tmp_u = self.rfft_object(u.values)
        tmp = self.params.nu * self.lap * tmp_u - self.params.c * self.ddx * tmp_u
        f.values[:] = np.real(self.irfft_object(tmp))

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
        tmp = self.rfft_object(rhs.values) / (1.0 - factor * (self.params.nu * self.lap - self.params.c * self.ddx))
        me.values[:] = np.real(self.irfft_object(tmp))

        return me
