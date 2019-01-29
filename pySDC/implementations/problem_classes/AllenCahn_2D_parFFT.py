import numpy as np

from mpi4py import MPI
from mpiFFT4py.line import R2C

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh


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

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=rhs_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = MPI.COMM_WORLD

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

        self.FFT = R2C(np.asarray(problem_params['nvars']), np.asarray((1, 1)), problem_params['comm'], "double")

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn2d_imex, self).__init__(init=self.FFT.real_shape(), dtype_u=dtype_u, dtype_f=dtype_f,
                                               params=problem_params)

        self.dx = self.params.L / self.params.nvars[0]  # could be useful for hooks, too.
        self.xvalues = np.array([i * self.dx - self.params.L / 2.0 for i in range(self.init[0])])
        self.yvalues = np.array([i * self.dx - self.params.L / 2.0 for i in range(self.init[1])])

        cinit = self.FFT.complex_shape()

        kx = np.zeros(cinit[0])
        ky = np.zeros(cinit[1])
        for i in range(0, int(cinit[0] / 2) + 1):
            kx[i] = 2 * np.pi / self.params.L * i
        for i in range(0, cinit[1]):
            ky[i] = 2 * np.pi / self.params.L * i
        for i in range(int(cinit[0] / 2) + 1, cinit[0]):
            kx[i] = 2 * np.pi / self.params.L * (-cinit[0] + i)

        self.lap = np.zeros((cinit[0], cinit[1]))
        for i in range(cinit[0]):
            for j in range(cinit[1]):
                self.lap[i, j] = -kx[i] ** 2 - ky[j] ** 2

        print(self.init, cinit)

        # self.U = np.zeros(self.FFT.real_shape(), dtype=self.FFT.real)
        self.U_hat = np.zeros(self.FFT.complex_shape(), dtype=self.FFT.complex)

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

        self.FFT.fft2(u.values, self.U_hat)
        tmp = self.lap * self.U_hat
        self.FFT.ifft2(tmp, f.impl.values[:])

        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
            f.expl.values = f.expl.values.reshape(self.init)
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
        self.FFT.fft2(rhs.values, self.U_hat)
        tmp = self.U_hat / (1.0 - factor * self.lap)
        self.FFT.ifft2(tmp, me.values[:])

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
            for i in range(self.init[0]):
                for j in range(self.init[1]):
                    r2 = self.xvalues[i] ** 2 + self.yvalues[j] ** 2
                    me.values[i, j] = np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))
        elif self.params.init_type == 'checkerboard':
            xv, yv = np.meshgrid(self.xvalues, self.xvalues)
            me.values[:, :] = np.sin(2.0 * np.pi * xv) * np.sin(2.0 * np.pi * yv)
        elif self.params.init_type == 'random':
            me.values[:, :] = np.random.uniform(-1, 1, self.init)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

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

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=rhs_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed to parent class)
            dtype_f: mesh data type wuth implicit and explicit parts (will be passed to parent class)
        """
        super(allencahn2d_imex_stab, self).__init__(problem_params=problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

        kx = np.zeros(self.init[0])
        ky = np.zeros(self.init[1] // 2 + 1)
        for i in range(0, int(self.init[0] / 2) + 1):
            kx[i] = 2 * np.pi / self.params.L * i
        for i in range(0, int(self.init[1] // 2) + 1):
            ky[i] = 2 * np.pi / self.params.L * i
        for i in range(int(self.init[0] / 2) + 1, self.init[0]):
            kx[i] = 2 * np.pi / self.params.L * (-self.init[0] + i)

        self.lap = np.zeros((self.init[0], self.init[1] // 2 + 1))
        for i in range(self.init[0]):
            for j in range(self.init[1] // 2 + 1):
                self.lap[i, j] = -kx[i] ** 2 - ky[j] ** 2 - 2.0 / self.params.eps ** 2

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
        tmp = self.lap * self.rfft_object(u.values)
        f.impl.values[:] = np.real(self.irfft_object(tmp))
        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu) + \
                2.0 / self.params.eps ** 2 * v
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

        tmp = self.rfft_object(rhs.values) / (1.0 - factor * self.lap)
        me.values[:] = np.real(self.irfft_object(tmp))

        return me
