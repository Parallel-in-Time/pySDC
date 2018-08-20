from __future__ import division
import numpy as np

import pyfftw

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class allencahn2d_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping

    Attributes:
        xvalues: grid points in space
        lap: spectral operator for Laplacian
        fft_object: planned FFT for forward transformation
        ifft_object: planned IFFT for backward transformation
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
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'L', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] != problem_params['nvars'][1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn2d_imex, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                               params=problem_params)

        dx = self.params.L / self.params.nvars[0]
        self.xvalues = np.array([i * dx - self.params.L / 2.0 for i in range(self.params.nvars[0])])

        kx = np.zeros(self.init[0])
        ky = np.zeros(self.init[1])
        for i in range(0, int(self.init[0] / 2) + 1):
            kx[i] = 2 * np.pi / self.params.L * i
        for i in range(0, int(self.init[1] / 2) + 1):
            ky[i] = 2 * np.pi / self.params.L * i
        for i in range(int(self.init[0] / 2) + 1, self.init[0]):
            kx[i] = 2 * np.pi / self.params.L * (-self.init[0] + i)
        for i in range(int(self.init[1] / 2) + 1, self.init[1]):
            ky[i] = 2 * np.pi / self.params.L * (-self.init[1] + i)

        self.lap = np.zeros(self.init)
        for i in range(self.init[0]):
            for j in range(self.init[1]):
                self.lap[i, j] = -kx[i] ** 2 - ky[j] ** 2

        # TODO: cleanup and move to real-valued FFT
        fft_in = pyfftw.empty_aligned(self.init, dtype='complex128')
        fft_out = pyfftw.empty_aligned(self.init, dtype='complex128')
        ifft_in = pyfftw.empty_aligned(self.init, dtype='complex128')
        ifft_out = pyfftw.empty_aligned(self.init, dtype='complex128')
        self.fft_object = pyfftw.FFTW(fft_in, fft_out, direction='FFTW_FORWARD', axes=(0, 1))
        self.ifft_object = pyfftw.FFTW(ifft_in, ifft_out, direction='FFTW_BACKWARD', axes=(0, 1))

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
        v = u.values.flatten()
        tmp = self.lap * self.fft_object(u.values)
        f.impl.values[:] = np.real(self.ifft_object(tmp))
        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
            f.expl.values = f.expl.values.reshape(self.params.nvars)
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

        tmp = self.fft_object(rhs.values) / (1.0 - factor * self.lap)
        me.values[:] = np.real(self.ifft_object(tmp))

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.params.init_type == 'circle':
            for i in range(self.params.nvars[0]):
                for j in range(self.params.nvars[1]):
                    r2 = self.xvalues[i] ** 2 + self.xvalues[j] ** 2
                    me.values[i, j] = np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))
        elif self.params.init_type == 'checkerboard':
            xv, yv = np.meshgrid(self.xvalues, self.xvalues)
            me.values[:, :] = np.sin(2.0 * np.pi * xv) * np.sin(2.0 * np.pi * yv)
        elif self.params.init_type == 'random':
            me.values[:, :] = np.random.uniform(-1, 1, self.init)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me
