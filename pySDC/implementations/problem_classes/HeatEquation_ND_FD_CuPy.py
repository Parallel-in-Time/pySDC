import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve, cg  # , gmres, minres

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh


# noinspection PyUnusedLocal
class heatNd_forced(ptype):
    """
    Example implementing the ND heat equation with periodic or Diriclet-Zero BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: FD discretization of the ND laplace operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=imex_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence

        if 'order' not in problem_params:
            problem_params['order'] = 2
        if 'stencil_type' not in problem_params:
            problem_params['stencil_type'] = 'center'
        if 'lintol' not in problem_params:
            problem_params['lintol'] = 1e-12
        if 'liniter' not in problem_params:
            problem_params['liniter'] = 10000
        if 'direct_solver' not in problem_params:
            problem_params['direct_solver'] = True

        essential_keys = ['nvars', 'nu', 'freq', 'order', 'lintol', 'liniter', 'direct_solver', 'bc']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if not (type(problem_params['nvars']) is tuple and type(problem_params['freq']) is tuple) and not (
            type(problem_params['nvars']) is int and type(problem_params['freq']) is int
        ):
            print(problem_params['nvars'], problem_params['freq'])
            raise ProblemError('Type of nvars and freq must be both either int or both tuple')

        if 'ndim' not in problem_params:
            if type(problem_params['nvars']) is int:
                problem_params['ndim'] = 1
            elif type(problem_params['nvars']) is tuple:
                problem_params['ndim'] = len(problem_params['nvars'])

        if problem_params['ndim'] > 3:
            raise ProblemError(f'can work with up to three dimensions, got {problem_params["ndim"]}')

        if type(problem_params['freq']) is tuple:
            for freq in problem_params['freq']:
                if freq % 2 != 0 and problem_params['bc'] == 'periodic':
                    raise ProblemError('need even number of frequencies due to periodic BCs')
        else:
            if problem_params['freq'] % 2 != 0 and problem_params['bc'] == 'periodic':
                raise ProblemError('need even number of frequencies due to periodic BCs')

        if type(problem_params['nvars']) is tuple:
            for nvars in problem_params['nvars']:
                if nvars % 2 != 0 and problem_params['bc'] == 'periodic':
                    raise ProblemError('the setup requires nvars = 2^p per dimension')
                if (nvars + 1) % 2 != 0 and problem_params['bc'] == 'dirichlet-zero':
                    raise ProblemError('setup requires nvars = 2^p - 1')
            if problem_params['nvars'][1:] != problem_params['nvars'][:-1]:
                raise ProblemError('need a square domain, got %s' % problem_params['nvars'])
        else:
            if problem_params['nvars'] % 2 != 0 and problem_params['bc'] == 'periodic':
                raise ProblemError('the setup requires nvars = 2^p per dimension')
            if (problem_params['nvars'] + 1) % 2 != 0 and problem_params['bc'] == 'dirichlet-zero':
                raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heatNd_forced, self).__init__(
            init=(problem_params['nvars'], None, cp.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        if self.params.ndim == 1:
            if type(self.params.nvars) is not tuple:
                self.params.nvars = (self.params.nvars,)
            if type(self.params.freq) is not tuple:
                self.params.freq = (self.params.freq,)

        # compute dx (equal in both dimensions) and get discretization matrix A
        if self.params.bc == 'periodic':
            self.dx = 1.0 / self.params.nvars[0]
            xvalues = cp.array([i * self.dx for i in range(self.params.nvars[0])])
        elif self.params.bc == 'dirichlet-zero':
            self.dx = 1.0 / (self.params.nvars[0] + 1)
            xvalues = cp.array([(i + 1) * self.dx for i in range(self.params.nvars[0])])
        else:
            raise ProblemError(f'Boundary conditions {self.params.bc} not implemented.')

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.params.order,
            type=self.params.stencil_type,
            dx=self.dx,
            size=self.params.nvars[0],
            dim=self.params.ndim,
            bc=self.params.bc,
            cupy=True,
        )
        self.A *= self.params.nu

        self.xv = cp.meshgrid(*[xvalues for _ in range(self.params.ndim)])
        self.Id = csp.eye(np.prod(self.params.nvars), format='csc')

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
        f.impl.values[:] = self.A.dot(u.values.flatten()).reshape(self.params.nvars)
        if self.params.ndim == 1:
            f.expl.values[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * (
                self.params.nu * np.pi**2 * sum([freq**2 for freq in self.params.freq]) * cp.cos(t) - cp.sin(t)
            )
        elif self.params.ndim == 2:
            f.expl.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * (self.params.nu * np.pi**2 * sum([freq**2 for freq in self.params.freq]) * cp.cos(t) - cp.sin(t))
            )
        elif self.params.ndim == 3:
            f.expl.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.params.freq[2] * self.xv[2])
                * (self.params.nu * np.pi**2 * sum([freq**2 for freq in self.params.freq]) * cp.cos(t) - cp.sin(t))
            )

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        if self.params.direct_solver:
            me.values[:] = spsolve(self.Id - factor * self.A, rhs.values.flatten()).reshape(self.params.nvars)
        else:
            me.values[:] = cg(
                self.Id - factor * self.A,
                rhs.values.flatten(),
                x0=u0.values.flatten(),
                tol=self.params.lintol,
                maxiter=self.params.liniter,
                atol=0,
            )[0].reshape(self.params.nvars)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)

        if self.params.ndim == 1:
            me.values[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * cp.cos(t)
        elif self.params.ndim == 2:
            me.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * cp.cos(t)
            )
        elif self.params.ndim == 3:
            me.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.params.freq[2] * self.xv[2])
                * cp.cos(t)
            )
        return me


class heatNd_unforced(heatNd_forced):
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(heatNd_unforced, self).__init__(problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

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
        f.values[:] = self.A.dot(u.flatten()).reshape(self.params.nvars)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)

        if self.params.ndim == 1:
            rho = (2.0 - 2.0 * cp.cos(np.pi * self.params.freq[0] * self.dx)) / self.dx**2
            me.values[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * cp.exp(-t * self.params.nu * rho)
        elif self.params.ndim == 2:
            rho = (2.0 - 2.0 * cp.cos(np.pi * self.params.freq[0] * self.dx)) / self.dx**2 + (
                2.0 - 2.0 * cp.cos(np.pi * self.params.freq[1] * self.dx)
            ) / self.dx**2

            me.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * cp.exp(-t * self.params.nu * rho)
            )
        elif self.params.ndim == 3:
            rho = (
                (2.0 - 2.0 * cp.cos(np.pi * self.params.freq[0] * self.dx)) / self.dx**2
                + (2.0 - 2.0 * cp.cos(np.pi * self.params.freq[1] * self.dx))
                + (2.0 - 2.0 * cp.cos(np.pi * self.params.freq[2] * self.dx)) / self.dx**2
            )
            me.values[:] = (
                cp.sin(np.pi * self.params.freq[0] * self.xv[0])
                * cp.sin(np.pi * self.params.freq[1] * self.xv[1])
                * cp.sin(np.pi * self.params.freq[2] * self.xv[2])
                * cp.exp(-t * self.params.nu * rho)
            )

        return me
