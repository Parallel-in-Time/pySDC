import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class monodomain2d_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping

    Attributes:
        xvalues: grid points in space
        dx: mesh width
        lap: spectral operator for Laplacian
        rfft_object: planned real FFT for forward transformation
        irfft_object: planned IFFT for backward transformation
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
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

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'a', 'kappa', 'rest', 'thresh', 'depol', 'init_type', 'eps']
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
        super(monodomain2d_imex, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.dx = self.params.L / self.params.nvars[0]  # could be useful for hooks, too.
        self.xvalues = np.array([i * self.dx - self.params.L / 2.0 for i in range(self.params.nvars[0])])

        kx = np.zeros(self.init[0][0])
        ky = np.zeros(self.init[0][1] // 2 + 1)

        kx[: int(self.init[0][0] / 2) + 1] = 2 * np.pi / self.params.L * np.arange(0, int(self.init[0][0] / 2) + 1)
        kx[int(self.init[0][0] / 2) + 1 :] = (
            2 * np.pi / self.params.L * np.arange(int(self.init[0][0] / 2) + 1 - self.init[0][0], 0)
        )
        ky[:] = 2 * np.pi / self.params.L * np.arange(0, self.init[0][1] // 2 + 1)

        xv, yv = np.meshgrid(kx, ky, indexing='ij')
        self.lap = -xv**2 - yv**2

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
        tmp = self.params.kappa * self.lap * np.fft.rfft2(u)
        f.impl[:] = np.fft.irfft2(tmp)
        f.expl[:] = -(self.params.a * (v - self.params.rest) * (v - self.params.thresh) * (v - self.params.depol)).reshape(self.params.nvars)
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

        tmp = np.fft.rfft2(rhs) / (1.0 - factor * self.params.kappa * self.lap)
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
            if self.params.init_type == 'tanh':
                xv, yv = np.meshgrid(self.xvalues, self.xvalues, indexing='ij')
                me[:, :] = 0.5 * (1 + np.tanh((self.params.radius - np.sqrt(xv**2 + yv**2)) / (np.sqrt(2) * self.params.eps)))
                me[:, :] = me[:, :] * self.params.depol + (1 - me[:, :]) * self.params.rest
            elif self.params.init_type == 'plateau':
                xv, yv = np.meshgrid(self.xvalues, self.xvalues, indexing='ij')
                me[:, :] = np.where(np.sqrt(xv**2 + yv**2) < self.params.radius * self.params.L, self.params.depol, self.params.rest)
                # me[:, :] = me[:, :] * self.params.depol + (1 - me[:, :]) * self.params.rest
            else:
                raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)
        else:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:, :] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        return me
