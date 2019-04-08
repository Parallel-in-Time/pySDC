import numpy as np

from pmesh.pm import ParticleMesh

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.playgrounds.pmesh.pmesh_datatype import pmesh_datatype, rhs_imex_pmesh


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

    def __init__(self, problem_params, dtype_u=pmesh_datatype, dtype_f=rhs_imex_pmesh):
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
            problem_params['comm'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'L', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])

        pm = ParticleMesh(BoxSize=1.0, Nmesh=list(problem_params['nvars']), dtype='f8', plan_method='measure',
                          comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn2d_imex, self).__init__(init=pm, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.dx = self.params.L / problem_params['nvars'][0]

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki ** 2 for ki in k)
            return -k2 * v

        f = self.dtype_f(self.init)
        f.impl.values = u.values.r2c().apply(Laplacian).c2r()
        if self.params.eps > 0:
            f.expl.values = 1.0 / self.params.eps ** 2 * u.values * (1.0 - u.values ** self.params.nu)
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

        def linear_solve(k, v):
            k2 = sum(ki ** 2 for ki in k)
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        me.values = rhs.values.r2c().apply(linear_solve).c2r()
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        def circle(i, v):
            r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
            r2 = sum(ri ** 2 for ri in r)
            return np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.params.init_type == 'circle':
            me.values.apply(circle, kind='index', out=Ellipsis)
        # elif self.params.init_type == 'checkerboard':
        #     xv, yv = np.meshgrid(self.xvalues, self.xvalues)
        #     me.values[:, :] = np.sin(2.0 * np.pi * xv) * np.sin(2.0 * np.pi * yv)
        # elif self.params.init_type == 'random':
        #     me.values[:, :] = np.random.uniform(-1, 1, self.init)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me


# class allencahn2d_imex_stab(allencahn2d_imex):
#     """
#     Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping with
#     stabilized splitting
#
#     Attributes:
#         xvalues: grid points in space
#         dx: mesh width
#         lap: spectral operator for Laplacian
#         rfft_object: planned real FFT for forward transformation
#         irfft_object: planned IFFT for backward transformation
#     """
#
#     def __init__(self, problem_params, dtype_u=mesh, dtype_f=rhs_imex_mesh):
#         """
#         Initialization routine
#
#         Args:
#             problem_params (dict): custom parameters for the example
#             dtype_u: mesh data type (will be passed to parent class)
#             dtype_f: mesh data type wuth implicit and explicit parts (will be passed to parent class)
#         """
#         super(allencahn2d_imex_stab, self).__init__(problem_params=problem_params, dtype_u=dtype_u, dtype_f=dtype_f)
#
#         self.lap -= 2.0 / self.params.eps ** 2
#
#     def eval_f(self, u, t):
#         """
#         Routine to evaluate the RHS
#
#         Args:
#             u (dtype_u): current values
#             t (float): current time
#
#         Returns:
#             dtype_f: the RHS
#         """
#
#         f = self.dtype_f(self.init)
#         v = u.values.flatten()
#         tmp = self.lap * self.rfft_object(u.values)
#         f.impl.values[:] = self.irfft_object(tmp)
#         if self.params.eps > 0:
#             f.expl.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu) + \
#                 2.0 / self.params.eps ** 2 * v
#             f.expl.values = f.expl.values.reshape(self.params.nvars)
#         return f
#
#     def solve_system(self, rhs, factor, u0, t):
#         """
#         Simple FFT solver for the diffusion part
#
#         Args:
#             rhs (dtype_f): right-hand side for the linear system
#             factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
#             u0 (dtype_u): initial guess for the iterative solver (not used here so far)
#             t (float): current time (e.g. for time-dependent BCs)
#
#         Returns:
#             dtype_u: solution as mesh
#         """
#
#         me = self.dtype_u(self.init)
#
#         tmp = self.rfft_object(rhs.values) / (1.0 - factor * self.lap)
#         me.values[:] = self.irfft_object(tmp)
#
#         return me
