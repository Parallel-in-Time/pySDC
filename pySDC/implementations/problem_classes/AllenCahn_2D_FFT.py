import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class allencahn2d_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping

    Attributes:
        xvalues: grid points in space
        dx: mesh width
        lap: spectral operator for Laplacian
        rfft_object: planned real FFT for forward transformation
        irfft_object: planned IFFT for backward transformation
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars, nu, eps, radius, L=1.0, init_type='circle'):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'eps', 'radius', 'L', 'init_type', localVars=locals(), readOnly=True
        )

        self.dx = self.L / self.nvars[0]  # could be useful for hooks, too.
        self.xvalues = np.array([i * self.dx - self.L / 2.0 for i in range(self.nvars[0])])

        kx = np.zeros(self.init[0][0])
        ky = np.zeros(self.init[0][1] // 2 + 1)

        kx[: int(self.init[0][0] / 2) + 1] = 2 * np.pi / self.L * np.arange(0, int(self.init[0][0] / 2) + 1)
        kx[int(self.init[0][0] / 2) + 1 :] = (
            2 * np.pi / self.L * np.arange(int(self.init[0][0] / 2) + 1 - self.init[0][0], 0)
        )
        ky[:] = 2 * np.pi / self.L * np.arange(0, self.init[0][1] // 2 + 1)

        xv, yv = np.meshgrid(kx, ky, indexing='ij')
        self.lap = -(xv**2) - yv**2

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
        v = u.flatten()
        tmp = self.lap * np.fft.rfft2(u)
        f.impl[:] = np.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)
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

        tmp = np.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = np.fft.irfft2(tmp)

        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time
            u_init (pySDC.implementations.problem_classes.allencahn2d_imex.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            if self.init_type == 'circle':
                xv, yv = np.meshgrid(self.xvalues, self.xvalues, indexing='ij')
                me[:, :] = np.tanh((self.radius - np.sqrt(xv**2 + yv**2)) / (np.sqrt(2) * self.eps))
            elif self.init_type == 'checkerboard':
                xv, yv = np.meshgrid(self.xvalues, self.xvalues)
                me[:, :] = np.sin(2.0 * np.pi * xv) * np.sin(2.0 * np.pi * yv)
            elif self.init_type == 'random':
                me[:, :] = np.random.uniform(-1, 1, self.init)
            else:
                raise NotImplementedError('type of initial value not implemented, got %s' % self.init_type)
        else:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:, :] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        return me


class allencahn2d_imex_stab(allencahn2d_imex):
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping with
    stabilized splitting

    Attributes:
        xvalues: grid points in space
        dx: mesh width
        lap: spectral operator for Laplacian
        rfft_object: planned real FFT for forward transformation
        irfft_object: planned IFFT for backward transformation
    """

    def __init__(self, nvars, nu, eps, radius, L=1.0, init_type='circle'):
        super().__init__(nvars, nu, eps, radius, L, init_type)
        self.lap -= 2.0 / self.eps**2

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
        v = u.flatten()
        tmp = self.lap * np.fft.rfft2(u)
        f.impl[:] = np.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu) + 2.0 / self.eps**2 * v).reshape(self.nvars)
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

        tmp = np.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = np.fft.irfft2(tmp)

        return me
